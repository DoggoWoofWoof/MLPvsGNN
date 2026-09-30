# MP-Approx level 7 (MP-ORACLE): the one-hop kernel with a query-free neighbour side

Declared in `configs/mp_approx_l7.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l7/record.json` (git-ignored), assembled 2026-09-30T09:04:54Z at `a5a53aa` from the host's per-dataset `read.json` files, without arithmetic.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

Level 7 of the MP-Approx ladder, on the MP-ORACLE track. The advisor's order after L6 reads "Only after that, build the deployable 2Wiki compiled-kernel MLP". Level 6 filed that level 5's kernel moments are per query and pool, not per node: psi and the values read the halo row n_u, whose first 258 columns depend on the query and the pool, and two edge attributes that depend on them. This file asks what the kernel keeps when its neighbour side reads only query-free inputs, before the deployable file fixes its form. As at every earlier level, the trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe or kernel is a retriever, a teacher or a feature.

**Placement: host-native.** Every probe this file compares, B0-mlp, L3-att and L5-kern included, was fitted and read on host_gpu_det as a new draw. No number here is set beside a number made on the laptop.

**The entries are in-pool.** Level 3's entries are each U_q row's in-pool neighbours (the first 64 per family, in the store's order). A query-free neighbour side makes psi and the values of every edge query-free. The moments over a fixed neighbourhood of a node, which would include nodes outside the pool, are not tested here.

## What was measured

- **L7 kernel**: level 5's factorised kernel. kappa_e^h = phi_h(v)^T psi_h(u, e), phi = softplus(W_phi [n_v | q_tilde]) (unchanged), psi = softplus(W_psi s_ue) and val_e = W_val s_ue, rank 32 per head, 4 heads of 16. alpha is kappa over its sum across the row's entries and its self entry. Level 3's readout reads [b_v | m_v].
- **The neighbour side s_ue**: for **L7-qi** (the primary), X_u R (n_u's columns 258-321) and eps_e's query-free columns: the family, rel_mask, dir_fwd, dir_bwd and rel_text (71). It drops the halo's 258 query-dependent columns, the pool-relative weight and rel_compat = cos(q, e_r). **L7-qw** keeps the pool-relative weight as well (not compile-once; a diagnostic). **L7-qi-mean** has uniform weights over the row's entries and its self entry, and L7-qi's values (the query-free neighbour mean). **L7-qi-list** is L7-qi under level 3's listwise objective.
- **Refitted** with the earlier levels' code, unchanged: B0-mlp (node-local) and L3-att (learned query-conditioned one-hop attention) from level 3, and L5-kern (the same kernel with the full neighbour side [n_u | eps_e]) from level 5. A laptop test holds the L7 kernel with every column kept equal to L5-kern, and holds L7-qi's moments C_v = sum psi val^T and c_v = sum psi unchanged when every query-dependent column changes.
- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** on full-pool metrics. L7_HIGH (>= 0.75, interval low >= 0.50), L7_LOW (<= 0.25, interval high <= 0.50), L7_MID otherwise.

## Readings

| dataset | queries | reading (L7-qi on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|
| metaqa | 2,400 | **L7_LOW** | -0.577 [-0.691, -0.477] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, L3-att); FIT_NOT_RANK (r, L5-kern); FIT_NOT_RANK (r, L7-qi); FIT_NOT_RANK (r, L7-qw); FIT_NOT_RANK (r, L7-qi-mean) | qi_below_kernel, qi_above_node_local, halo_carries, objective_adds |
| 2wiki | 6,290 | **L7_LOW** | 0.133 [0.079, 0.183] | recall@5, full_coverage@5 | none | qi_below_kernel, qi_above_node_local, halo_carries, objective_adds |
| squad | 5,841 | **NOT_READ** | n/a | none | SEED_BOUND | none |

Across datasets, from the per-dataset entries only (no number pooled): **compile_once_supported** no; **compile_once_partial** yes; **compile_once_not_supported** no; **weight_needed** no; **kernel_reference_held** yes; **negative_control_as_expected** yes.

What the deployable file reads here, as filed before any number: What the later deployable 2wiki file reads here. None of it opens that file. compile_once_supported: its kernel may use node moments compiled once from a query-free neighbour side. compile_once_partial: it declares both forms, the per-query form as the reference, and measures the retrieval cost of the compiled form directly. compile_once_not_supported: it uses moments compiled per query. weight_needed: its query-free neighbour side carries the store's raw edge weight, which is query-free and could not be tested here. In every case the fixed neighbourhood (corrections_to_the_proposal.two_conditions_for_compile_once, (b)) is that file's to test.

### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)

| probe | metaqa | 2wiki | squad |
|---|---|---|---|
| B0-mlp | -0.650 [-0.769, -0.549] (L7_LOW) | 0.055 [-0.002, 0.109] (L7_LOW) | n/a (NOT_READ) |
| L3-att | -0.076 [-0.161, 0.002] (L7_LOW) | 0.799 [0.759, 0.838] (L7_HIGH) | n/a (NOT_READ) |
| L5-kern | -0.358 [-0.466, -0.268] (L7_LOW) | 0.746 [0.706, 0.789] (L7_MID) | n/a (NOT_READ) |
| L7-qi | -0.577 [-0.691, -0.477] (L7_LOW) | 0.133 [0.079, 0.183] (L7_LOW) | n/a (NOT_READ) |
| L7-qw | -0.607 [-0.729, -0.507] (L7_LOW) | 0.133 [0.083, 0.183] (L7_LOW) | n/a (NOT_READ) |
| L7-qi-mean | -0.567 [-0.684, -0.469] (L7_LOW) | 0.154 [0.100, 0.206] (L7_LOW) | n/a (NOT_READ) |
| L7-qi-list | -0.168 [-0.245, -0.103] (L7_LOW) | 0.325 [0.278, 0.371] (L7_MID) | n/a (NOT_READ) |
| ref:other_seed | 1.064 [1.044, 1.086] (L7_HIGH) | 0.867 [0.837, 0.893] (L7_HIGH) | n/a (NOT_READ) |

### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)

| contrast | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| deploy_cost | L7-qi - L5-kern, r | -0.219 [-0.282, -0.156] | -0.613 [-0.670, -0.559] | not read |
| halo_cost | L7-qw - L5-kern, r | -0.249 [-0.318, -0.188] | -0.613 [-0.667, -0.560] | not read |
| weight_cost | L7-qi - L7-qw, r | 0.030 [-0.012, 0.074] | 0.000 [-0.030, 0.027] | not read |
| qi_vs_attention | L7-qi - L3-att, r | -0.501 [-0.583, -0.432] | -0.665 [-0.721, -0.613] | not read |
| qi_over_node_local | L7-qi - B0-mlp, r | 0.073 [0.013, 0.134] | 0.078 [0.038, 0.118] | not read |
| query_conditioning | L7-qi - L7-qi-mean, r | -0.010 [-0.059, 0.035] | -0.021 [-0.055, 0.013] | not read |
| objective | L7-qi-list - L7-qi, r | 0.410 [0.336, 0.488] | 0.192 [0.153, 0.231] | not read |
| kernel_vs_attention | L5-kern - L3-att, r | -0.282 [-0.347, -0.225] | -0.053 [-0.081, -0.025] | not read |
| edge_qi_vs_attention | L7-qi - L3-att, e | -0.059 [-0.069, -0.047] | -0.194 [-0.208, -0.179] | not read |

### Shares (descriptive; read only where the denominator's interval lies above 0)

| share | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| qi_share | (L7-qi - B0-mlp) / (L5-kern - B0-mlp), r | 0.250 [0.049, 0.430] | 0.113 [0.058, 0.167] | not read |
| qi_attention_share | (L7-qi - B0-mlp) / (L3-att - B0-mlp), r | 0.127 [0.023, 0.223] | 0.105 [0.054, 0.157] | not read |

What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):

- **halo_carries**: the halo_cost interval below 0 -- the halo's query-dependent columns (and rel_compat) carry recovery that X_u R and the query-free edge columns do not
- **objective_adds**: the objective interval above 0 -- the objective limits recovery, not only the operator
- **qi_above_node_local**: the qi_over_node_local interval above 0
- **qi_below_kernel**: the deploy_cost interval below 0 -- the query-free neighbour side recovers less than the full one

### Level 5, as filed (bands and interpretation entries only)

| dataset | level 5 reading (L5-mom) | L5-kern band on r | level 5 interpretation |
|---|---|---|---|
| metaqa | L5_LOW | L5_LOW | moments_below_attention, kernel_below_attention, objective_adds |
| 2wiki | L5_MID | L5_MID | moments_add_over_mean, moments_below_attention, kernel_below_attention, objective_adds |
| squad | NOT_READ | NOT_READ | none |

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
| B0-mlp | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L7_LOW |
| L3-att | 0.690 | 0.781 | 0.649 | 0.626 | 0.602 | 0.025 | 0.034 | -0.287 | -0.076 [-0.161, 0.002] | L7_LOW |
| L5-kern | 0.660 | 0.762 | 0.638 | 0.587 | 0.579 | -0.250 | -0.235 | -0.590 | -0.358 [-0.466, -0.268] | L7_LOW |
| L7-qi | 0.578 | 0.707 | 0.582 | 0.510 | 0.546 | -0.546 | -0.476 | -0.709 | -0.577 [-0.691, -0.477] | L7_LOW |
| L7-qw | 0.579 | 0.707 | 0.579 | 0.509 | 0.546 | -0.541 | -0.462 | -0.818 | -0.607 [-0.729, -0.507] | L7_LOW |
| L7-qi-mean | 0.578 | 0.701 | 0.567 | 0.495 | 0.548 | -0.506 | -0.406 | -0.791 | -0.567 [-0.684, -0.469] | L7_LOW |
| L7-qi-list | 0.232 | 0.457 | 0.397 | 0.489 | 0.535 | -0.129 | -0.032 | -0.342 | -0.168 [-0.245, -0.103] | L7_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L7_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L7_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.790 | 0.856 | 0.750 | 0.893 | 0.520 | 0.733 | 0.680 | 0.600 | 0.671 [0.655, 0.687] | L7_MID |
| L3-att | 0.842 | 0.886 | 0.776 | 0.918 | 0.572 | 0.809 | 0.759 | 0.692 | 0.753 [0.738, 0.769] | L7_HIGH |
| L7-qi | 0.801 | 0.864 | 0.761 | 0.902 | 0.533 | 0.748 | 0.694 | 0.641 | 0.694 [0.677, 0.712] | L7_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L7_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L7_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)

| stratum | queries | readable | B0-mlp rho_bar | L3-att rho_bar | L5-kern rho_bar | L7-qi rho_bar | L7-qw rho_bar | L7-qi-mean rho_bar | L7-qi-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.039 [-1.842, -0.582] | 0.228 [-0.167, 0.521] | 0.101 [-0.315, 0.409] | -0.229 [-0.743, 0.095] | -0.202 [-0.669, 0.097] | -0.252 [-0.770, 0.047] | 0.034 [-0.384, 0.305] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -1.049 [-1.466, -0.759] | -0.650 [-1.022, -0.402] | -1.046 [-1.540, -0.719] | -1.202 [-1.663, -0.881] | -1.340 [-1.854, -0.991] | -1.056 [-1.476, -0.755] | -0.257 [-0.481, -0.097] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.489 [-0.602, -0.386] | 0.045 [-0.038, 0.120] | -0.225 [-0.327, -0.135] | -0.448 [-0.561, -0.355] | -0.453 [-0.566, -0.361] | -0.473 [-0.589, -0.376] | -0.171 [-0.253, -0.103] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.619 [-0.798, -0.474] | 0.056 [-0.070, 0.156] | -0.227 [-0.373, -0.112] | -0.445 [-0.601, -0.316] | -0.422 [-0.580, -0.301] | -0.444 [-0.602, -0.318] | -0.181 [-0.305, -0.083] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -0.577 [-0.797, -0.418] | -0.193 [-0.366, -0.065] | -0.452 [-0.660, -0.297] | -0.631 [-0.848, -0.470] | -0.733 [-0.961, -0.570] | -0.564 [-0.766, -0.416] | -0.097 [-0.208, -0.003] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -0.898 [-1.256, -0.631] | -0.189 [-0.462, 0.006] | -0.517 [-0.826, -0.266] | -0.829 [-1.176, -0.567] | -0.844 [-1.204, -0.591] | -0.916 [-1.313, -0.644] | -0.294 [-0.498, -0.137] | 1.013 [0.971, 1.057] |

| stratum | deploy_cost | qi_vs_attention | query_conditioning |
|---|---|---|---|
| hop=1 | -0.331 [-0.652, -0.069] | -0.458 [-0.804, -0.211] | 0.023 [-0.172, 0.243] |
| hop=2 | -0.156 [-0.362, 0.043] | -0.552 [-0.815, -0.349] | -0.146 [-0.328, -0.013] |
| hop=3 | -0.223 [-0.291, -0.160] | -0.493 [-0.580, -0.419] | 0.025 [-0.025, 0.071] |
| first_support_STRUCT=no_gold_in_pool | not read | not read | not read |
| first_support_STRUCT=h1 | -0.218 [-0.314, -0.138] | -0.500 [-0.616, -0.405] | -0.000 [-0.066, 0.066] |
| first_support_STRUCT=h2 | -0.179 [-0.294, -0.065] | -0.438 [-0.590, -0.315] | -0.067 [-0.162, 0.006] |
| first_support_STRUCT=h3 | -0.312 [-0.477, -0.168] | -0.641 [-0.858, -0.463] | 0.087 [-0.029, 0.225] |

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
| B0-mlp | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L7_LOW |
| L3-att | 0.645 | 0.753 | 0.596 | 0.665 | 0.816 | 0.831 | 0.767 | -0.432 (nr) | 0.799 [0.759, 0.838] | L7_HIGH |
| L5-kern | 0.636 | 0.748 | 0.594 | 0.662 | 0.813 | 0.772 | 0.720 | -0.735 (nr) | 0.746 [0.706, 0.789] | L7_MID |
| L7-qi | 0.482 | 0.661 | 0.514 | 0.511 | 0.765 | 0.179 | 0.087 | -1.568 (nr) | 0.133 [0.079, 0.183] | L7_LOW |
| L7-qw | 0.483 | 0.662 | 0.516 | 0.513 | 0.765 | 0.181 | 0.085 | -1.485 (nr) | 0.133 [0.083, 0.183] | L7_LOW |
| L7-qi-mean | 0.485 | 0.662 | 0.511 | 0.509 | 0.765 | 0.204 | 0.104 | -1.636 (nr) | 0.154 [0.100, 0.206] | L7_LOW |
| L7-qi-list | 0.336 | 0.580 | 0.470 | 0.515 | 0.762 | 0.365 | 0.285 | 0.311 (nr) | 0.325 [0.278, 0.371] | L7_MID |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L7_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L7_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.573 | 0.707 | 0.604 | 0.692 | 0.753 | 0.542 | 0.495 | 2.598 (nr) | 0.518 [0.502, 0.534] | L7_MID |
| L3-att | 0.782 | 0.809 | 0.712 | 0.853 | 0.834 | 0.841 | 0.792 | 1.937 (nr) | 0.816 [0.802, 0.829] | L7_HIGH |
| L7-qi | 0.632 | 0.734 | 0.621 | 0.753 | 0.777 | 0.652 | 0.594 | 2.495 (nr) | 0.623 [0.607, 0.637] | L7_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L7_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L7_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)

| stratum | queries | readable | B0-mlp rho_bar | L3-att rho_bar | L5-kern rho_bar | L7-qi rho_bar | L7-qw rho_bar | L7-qi-mean rho_bar | L7-qi-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | 0.193 [0.128, 0.255] | 0.948 [0.898, 0.997] | 0.876 [0.829, 0.925] | 0.283 [0.220, 0.341] | 0.289 [0.227, 0.349] | 0.329 [0.270, 0.392] | 0.442 [0.386, 0.499] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.232 [-0.343, -0.141] | 0.491 [0.420, 0.562] | 0.476 [0.402, 0.549] | -0.179 [-0.279, -0.085] | -0.190 [-0.295, -0.098] | -0.212 [-0.319, -0.120] | 0.084 [-0.002, 0.169] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.567 [-0.025, 1.209] | 0.238 [-0.548, 0.809] | 0.147 [-0.733, 0.683] | 0.567 [-0.036, 1.231] | 0.373 [-0.215, 0.992] | 0.228 [-0.600, 0.854] | 0.425 [-0.126, 0.914] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | 0.033 [-0.024, 0.087] | 0.805 [0.764, 0.846] | 0.755 [0.713, 0.800] | 0.109 [0.052, 0.159] | 0.109 [0.057, 0.161] | 0.139 [0.084, 0.191] | 0.319 [0.272, 0.366] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.476 [0.267, 0.718] | 0.766 [0.528, 0.989] | 0.633 [0.357, 0.886] | 0.628 [0.425, 0.872] | 0.603 [0.398, 0.854] | 0.501 [0.316, 0.696] | 0.459 [0.243, 0.696] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| stratum | deploy_cost | qi_vs_attention | query_conditioning |
|---|---|---|---|
| gold_total=2 | -0.593 [-0.664, -0.531] | -0.665 [-0.730, -0.596] | -0.046 [-0.087, -0.005] |
| gold_total>=3 | -0.655 [-0.753, -0.569] | -0.670 [-0.763, -0.591] | 0.033 [-0.030, 0.092] |
| first_support_STRUCT=no_gold_in_pool | not read | not read | not read |
| first_support_STRUCT=none | 0.420 [0.123, 1.191] | 0.328 [-0.019, 1.030] | 0.338 [-0.094, 1.183] |
| first_support_STRUCT=h1 | -0.647 [-0.706, -0.591] | -0.696 [-0.753, -0.640] | -0.030 [-0.064, 0.004] |
| first_support_STRUCT=h2 | -0.005 [-0.232, 0.264] | -0.138 [-0.333, 0.069] | 0.127 [-0.036, 0.339] |
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
| L5-kern | 0.525 | 0.674 | 0.513 | 0.444 | 0.813 | -0.184 (nr) | -0.184 (nr) | -0.019 (nr) | n/a | NOT_READ |
| L7-qi | 0.537 | 0.685 | 0.532 | 0.468 | 0.818 | -0.224 (nr) | -0.224 (nr) | -0.082 (nr) | n/a | NOT_READ |
| L7-qw | 0.536 | 0.684 | 0.526 | 0.467 | 0.817 | -0.303 (nr) | -0.303 (nr) | 0.031 (nr) | n/a | NOT_READ |
| L7-qi-mean | 0.492 | 0.645 | 0.481 | 0.405 | 0.803 | -0.118 (nr) | -0.118 (nr) | -0.170 (nr) | n/a | NOT_READ |
| L7-qi-list | 0.453 | 0.622 | 0.471 | 0.456 | 0.806 | -0.066 (nr) | -0.066 (nr) | 0.346 (nr) | n/a | NOT_READ |
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
| L7-qi | 0.498 | 0.671 | 0.561 | 0.503 | 0.954 | 0.778 (nr) | 0.778 (nr) | 1.238 (nr) | n/a | NOT_READ |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)

| stratum | queries | readable | B0-mlp rho_bar | L3-att rho_bar | L5-kern rho_bar | L7-qi rho_bar | L7-qw rho_bar | L7-qi-mean rho_bar | L7-qi-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| stratum | deploy_cost | qi_vs_attention | query_conditioning |
|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | not read | not read | not read |
| first_support_STRUCT=none | not read | not read | not read |
| first_support_STRUCT=h1 | not read | not read | not read |
| first_support_STRUCT=h2 | not read | not read | not read |
| first_support_STRUCT=h3 | not read | not read | not read |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity within U_q; it never describes a deployable model, and no probe output or kernel enters any retriever, feature, teacher or selection.
- The entries are in-pool. No reading here says that node moments compiled once over a fixed neighbourhood, which would include neighbours outside the pool, keep the recovery. That is the deployable file's to test, with the package.
- L7-qi drops the pool-relative weight because level 3's sidecar holds no other. The store's raw edge weight is query-free, and a query-free kernel that reads it is not tested here.
- The deployable 2wiki compiled-kernel model would be a competitor trained without GNN outputs, and it needs its own declaration under the QLS-U contract. This file does not open it, nor the latency retiming.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).

## Reproducibility, placement and compute

| dataset | probes (min) | read (min) | device | driver | determinism warnings | level 3 mirror.json sha256 (host) | repeat | refits bit-identical |
|---|---:|---:|---|---|---:|---|---|---|
| metaqa | 37 | 0.9 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `049d5cd2c3719895` | n/a | 5 of 5 |
| 2wiki | 47 | 1.9 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `4f2db84a5a1b0d80` | unit bit-identical | 5 of 5 |
| squad | 43 | 1.8 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `de7c1917942862d2` | n/a | 5 of 5 |

Parameters per arm: L7-qi 133,633, L7-qw 133,825, L7-qi-mean 66,689, L7-qi-list 133,633, L5-kern 183,553.

Placement: `host_gpu_det` = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": false, "env": "mpr-cu128@62fc45e9e1ba"}, applied by level 3's host_placement in every host process, with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified level 3's mirror.json against every level 0 and level 3 file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was filed and checked against the committed files at the file stage. Deviations: none.
