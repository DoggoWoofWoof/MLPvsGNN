# MP-Approx level 3 (MP-ORACLE): one-hop query-conditioned attention over fixed compiled rows

Declared in `configs/mp_approx_l3.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l3/record.json` (git-ignored), assembled 2026-09-29T19:17:03Z at `b8c64b2` from the host's per-dataset `read.json` files, without arithmetic.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

This is level 3 of the MP-Approx ladder, on the MP-ORACLE track. Level 1's interpretation entry l1_low_tokens_flat fired on metaqa and 2wiki and pointed past levels 1 and 2 to query-conditioned neighbourhood weighting; level 2 stays unopened. As at levels 0 and 1, the trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe, entry or halo row is a retriever, a teacher or a feature.

**Placement: host-native on host_gpu_det** (configs/gpu_task_qualification.yaml: TASK_EQUIVALENT and TRAINING_REPRODUCIBLE). Every probe this file compares, B0-mlp included, was fitted and read on the host GPU as a new draw. No number here is set beside a number made on the laptop: levels 0 and 1 are cited below by their filed bands and interpretation entries only.

## What was measured

- **Entries**: every message edge u -> v of the pool graph the GNN's cell reads (FULL: structural, ner and knn, in packed order, duplicates kept) whose target v is a U_q row, taken from the frozen packer's single-query batch. Each carries its family, its five edge attributes (float16) and up to four relation-text slots. Each row also gets one self entry.
- **Halo rows** (fixed): for every U_q row and every in-neighbour, B0 = sign(x) log1p(|x|) (129) | the pool z-score of x (129) | X R (64, level 1's JL matrix), float16. No learned neighbour state is read, and there is one hop.
- **L3-att**: score_e^h = a_h . LeakyReLU_0.2(W_u n_u + W_v [n_v | q_tilde] + (W_q q_tilde) * (W_e1 eps_e) + W_e2 eps_e), 4 heads of 16, softmax over the row's entries and its self entry; m_v = the heads' sum of alpha times W_val [n_u | eps_e]; readout 2x128 MLP on [b_v | m_v]. **L3-mean** is the same probe with every score held at 0 (uniform weighting of the same entries, features, values and readout). **B0-mlp** is level 0's MLP on b_v alone, refitted here.
- **Objectives**: MSE (level 0's) and LIST = KL(softmax(z(G_k)) || softmax(z(T_k) + rhat)) per query over U_q, temperature 1; its optimum is rhat = r_k up to a constant, so it fits the same target with the errors weighted toward the top of the GNN's ranking. No gold enters any loss.
- **Targets, recovery and U_q** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k). rho_M = (M(z(T_k) + rhat_k) - M(T_k)) / (M(G_k) - M(T_k)) for recall@5, full_coverage@5 and hit@1, read only where the gap's interval lies above 0; rho_bar is their mean. Metrics within U_q are an **upper bound** on full-pool metrics, so a LOW reading is robust and a HIGH one is not a full-pool claim.

Bands of rho_bar (level 0's thresholds): L3_HIGH (>= 0.75, interval low >= 0.50), L3_LOW (<= 0.25, interval high <= 0.50), L3_MID otherwise, NOT_READ with no readable metric. The dataset's reading is the band of L3-att on r_k.

## Readings

| dataset | queries | reading (L3-att on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|
| metaqa | 2,400 | **L3_LOW** | -0.076 [-0.161, 0.002] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, L3-mean); FIT_NOT_RANK (r, L3-att) | neighbourhood_adds, objective_adds, edge_effect_weighted |
| 2wiki | 6,290 | **L3_HIGH** | 0.799 [0.759, 0.838] | recall@5, full_coverage@5 | none | l3_high, l3_weighting_adds, neighbourhood_adds, objective_adds, edge_effect_weighted |
| squad | 5,841 | **NOT_READ** | n/a | none | SEED_BOUND | none |

### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)

| dataset | query_weighting (L3-att - L3-mean, r) | neighbourhood (L3-mean - B0-mlp, r) | attention_over_node_local (L3-att - B0-mlp, r) | objective_node_local (B0-list - B0-mlp, r) | objective_attention (L3-list - L3-att, r) | edge_weighting (L3-att - L3-mean, e) | edge_neighbourhood (L3-mean - B0-mlp, e) |
|---|---|---|---|---|---|---|---|
| metaqa | 0.442 [0.380, 0.517] | 0.132 [0.073, 0.195] | 0.574 [0.497, 0.661] | 0.394 [0.318, 0.475] | 0.315 [0.256, 0.380] | 0.049 [0.039, 0.059] | 0.033 [0.023, 0.043] |
| 2wiki | 0.237 [0.199, 0.276] | 0.507 [0.456, 0.560] | 0.744 [0.689, 0.803] | 0.189 [0.156, 0.227] | 0.103 [0.074, 0.134] | 0.117 [0.105, 0.129] | 0.181 [0.168, 0.195] |
| squad | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):

- **edge_effect_weighted**: the band of L3-att on e is L3_HIGH, or edge_weighting's interval is above 0: the GNN's own edge effect is read by one-hop weighting of fixed rows
- **l3_high**: primary L3_HIGH: one-hop query-conditioned weighting of fixed compiled features predicts the GNN's ranking effect within U_q on that dataset (an upper bound, not a full-pool claim). Being the ceiling of the compiled-moment family, it licenses the compiled-moment arm (phi(q,v)^T psi(u) with compiled sum psi(u)(W h_u)^T and sum psi(u)) as the next file.
- **l3_weighting_adds**: the query_weighting interval above 0 while the primary is L3_MID or L3_HIGH: the learned weighting, not the neighbourhood alone, carries part of the effect. With L3_MID it also licenses the compiled-moment arm, whose question is how much of the ceiling survives compilation.
- **neighbourhood_adds**: the neighbourhood interval above 0 -- fixed neighbour rows read uniformly already add over node-local alone
- **objective_adds**: either objective interval above 0: the probe's objective limits recovery, not only its features. The MSE readings of earlier levels then understate their features' capacity; re-reading them under LIST needs its own file, host-native.

### Levels 0 and 1, as filed (bands and interpretation entries only; their numbers were made on the laptop)

| dataset | level 0 reading | level 0 interpretation | level 1 reading | level 1 interpretation |
|---|---|---|---|---|
| metaqa | L0_LOW | primary_low_vectors_add, b1_low, message_not_retrieval | L1_LOW | l1_low_tokens_flat |
| 2wiki | L0_LOW | primary_low_vectors_add, b1_low, message_not_retrieval | L1_LOW | l1_low_tokens_flat |
| squad | NOT_READ | none | NOT_READ | message_tokens_add |

## metaqa

2,400 V2_GATE queries (the declared subsample: 800 per hop of the 19,738), 165,893 U_q rows (mean |U_q| 69.1). Compiled: 1,419,019 halo rows (591.3 per query) and 1,967,659 entries (819.9 per query: structural 1,736,405, ner 26,602, knn 204,652); 449 U_q rows have no in-edge and read only their self entry; 1,736,405 entries carry relation text (9 relations).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.061 | [0.055, 0.068] | yes |
| full_coverage@5 | 0.065 | [0.055, 0.074] | yes |
| hit@1 | 0.127 | [0.114, 0.140] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L3_LOW |
| B0-list | 0.189 | 0.424 | 0.374 | 0.458 | 0.530 | -0.160 | -0.058 | -0.551 | -0.256 [-0.342, -0.184] | L3_LOW |
| L3-mean | 0.611 | 0.722 | 0.591 | 0.516 | 0.561 | -0.452 | -0.387 | -0.716 | -0.518 [-0.633, -0.423] | L3_LOW |
| L3-att | 0.690 | 0.781 | 0.649 | 0.626 | 0.602 | 0.025 | 0.034 | -0.287 | -0.076 [-0.161, 0.002] | L3_LOW |
| L3-list | 0.391 | 0.590 | 0.502 | 0.663 | 0.581 | 0.330 | 0.325 | 0.062 | 0.239 [0.167, 0.303] | L3_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L3_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L3_HIGH |

Top-5 overlap of T_k itself with G_k: 0.529 [0.520, 0.538]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.790 | 0.856 | 0.750 | 0.893 | 0.520 | 0.733 | 0.680 | 0.600 | 0.671 [0.655, 0.687] | L3_MID |
| L3-mean | 0.812 | 0.869 | 0.764 | 0.902 | 0.542 | 0.761 | 0.713 | 0.637 | 0.704 [0.687, 0.720] | L3_MID |
| L3-att | 0.842 | 0.886 | 0.776 | 0.918 | 0.572 | 0.809 | 0.759 | 0.692 | 0.753 [0.738, 0.769] | L3_HIGH |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L3_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L3_HIGH |

Top-5 overlap of G0_k itself with G_k: 0.246 [0.237, 0.254]. (nr) = metric not readable on this dataset.

Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | B0-list rho_bar | L3-mean rho_bar | L3-att rho_bar | L3-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.039 [-1.842, -0.582] | -0.358 [-0.909, -0.037] | 0.061 [-0.366, 0.348] | 0.228 [-0.167, 0.521] | 0.291 [-0.113, 0.547] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -1.049 [-1.466, -0.759] | -0.254 [-0.478, -0.068] | -0.996 [-1.401, -0.703] | -0.650 [-1.022, -0.402] | 0.054 [-0.152, 0.213] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.489 [-0.602, -0.386] | -0.244 [-0.325, -0.176] | -0.459 [-0.580, -0.365] | 0.045 [-0.038, 0.120] | 0.283 [0.203, 0.355] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.619 [-0.798, -0.474] | -0.340 [-0.487, -0.225] | -0.349 [-0.507, -0.223] | 0.056 [-0.070, 0.156] | 0.152 [0.025, 0.255] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -0.577 [-0.797, -0.418] | -0.092 [-0.206, 0.006] | -0.570 [-0.787, -0.422] | -0.193 [-0.366, -0.065] | 0.280 [0.167, 0.366] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -0.898 [-1.256, -0.631] | -0.396 [-0.592, -0.244] | -0.879 [-1.235, -0.610] | -0.189 [-0.462, 0.006] | 0.384 [0.221, 0.544] | 1.013 [0.971, 1.057] |

## 2wiki

6,290 V2_GATE queries (all of them), 245,055 U_q rows (mean |U_q| 39.0). Compiled: 527,882 halo rows (83.9 per query) and 1,580,750 entries (251.3 per query: structural 842,608, ner 480,272, knn 257,870); 31,654 U_q rows have no in-edge and read only their self entry; 0 entries carry relation text (0 relations).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.048 | [0.045, 0.051] | yes |
| full_coverage@5 | 0.104 | [0.097, 0.111] | yes |
| hit@1 | -0.007 | [-0.012, -0.002] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L3_LOW |
| B0-list | 0.322 | 0.555 | 0.430 | 0.480 | 0.756 | 0.277 | 0.212 | -1.402 (nr) | 0.244 [0.199, 0.293] | L3_LOW |
| L3-mean | 0.584 | 0.718 | 0.564 | 0.595 | 0.794 | 0.589 | 0.535 | -1.424 (nr) | 0.562 [0.518, 0.606] | L3_MID |
| L3-att | 0.645 | 0.753 | 0.596 | 0.665 | 0.816 | 0.831 | 0.767 | -0.432 (nr) | 0.799 [0.759, 0.838] | L3_HIGH |
| L3-list | 0.559 | 0.705 | 0.565 | 0.660 | 0.815 | 0.921 | 0.883 | 0.242 (nr) | 0.902 [0.862, 0.942] | L3_HIGH |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L3_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L3_HIGH |

Top-5 overlap of T_k itself with G_k: 0.704 [0.700, 0.707]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.573 | 0.707 | 0.604 | 0.692 | 0.753 | 0.542 | 0.495 | 2.598 (nr) | 0.518 [0.502, 0.534] | L3_MID |
| L3-mean | 0.703 | 0.779 | 0.685 | 0.796 | 0.800 | 0.726 | 0.673 | 2.397 (nr) | 0.700 [0.684, 0.715] | L3_MID |
| L3-att | 0.782 | 0.809 | 0.712 | 0.853 | 0.834 | 0.841 | 0.792 | 1.937 (nr) | 0.816 [0.802, 0.829] | L3_HIGH |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L3_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L3_HIGH |

Top-5 overlap of G0_k itself with G_k: 0.652 [0.647, 0.657]. (nr) = metric not readable on this dataset.

Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | B0-list rho_bar | L3-mean rho_bar | L3-att rho_bar | L3-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | 0.193 [0.128, 0.255] | 0.353 [0.296, 0.412] | 0.683 [0.629, 0.736] | 0.948 [0.898, 0.997] | 1.003 [0.956, 1.056] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.232 [-0.343, -0.141] | 0.019 [-0.066, 0.096] | 0.311 [0.233, 0.384] | 0.491 [0.420, 0.562] | 0.692 [0.629, 0.753] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.567 [-0.025, 1.209] | 0.670 [0.154, 1.320] | 0.568 [0.062, 1.236] | 0.238 [-0.548, 0.809] | 0.608 [0.083, 1.292] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | 0.033 [-0.024, 0.087] | 0.235 [0.190, 0.286] | 0.559 [0.516, 0.605] | 0.805 [0.764, 0.846] | 0.913 [0.875, 0.953] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.476 [0.267, 0.718] | 0.360 [0.158, 0.582] | 0.649 [0.397, 0.931] | 0.766 [0.528, 0.989] | 0.660 [0.436, 0.870] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a | n/a | n/a |

## squad

5,841 V2_GATE queries (all of them; the control), 176,880 U_q rows (mean |U_q| 30.3). Compiled: 239,982 halo rows (41.1 per query) and 1,905,184 entries (326.2 per query: structural 1,050,095, ner 506,373, knn 348,716); 20,514 U_q rows have no in-edge and read only their self entry; 0 entries carry relation text (0 relations).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.004 | [-0.007, -0.002] | no |
| full_coverage@5 | -0.004 | [-0.007, -0.002] | no |
| hit@1 | -0.009 | [-0.013, -0.005] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.443 | 0.600 | 0.421 | 0.348 | 0.793 | -0.342 (nr) | -0.342 (nr) | -0.252 (nr) | n/a | NOT_READ |
| B0-list | 0.386 | 0.558 | 0.402 | 0.356 | 0.793 | -0.289 (nr) | -0.289 (nr) | -0.069 (nr) | n/a | NOT_READ |
| L3-mean | 0.489 | 0.639 | 0.471 | 0.393 | 0.803 | -0.066 (nr) | -0.066 (nr) | -0.101 (nr) | n/a | NOT_READ |
| L3-att | 0.517 | 0.666 | 0.505 | 0.431 | 0.811 | 0.000 (nr) | 0.000 (nr) | -0.031 (nr) | n/a | NOT_READ |
| L3-list | 0.428 | 0.596 | 0.442 | 0.420 | 0.801 | 0.158 (nr) | 0.158 (nr) | 0.013 (nr) | n/a | NOT_READ |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.746 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | -0.245 | 0.167 | 0.278 | 0.316 | 0.724 | 2.105 (nr) | 2.105 (nr) | 1.836 (nr) | n/a | NOT_READ |

Top-5 overlap of T_k itself with G_k: 0.746 [0.743, 0.749]. (nr) = metric not readable on this dataset.

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
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Top-5 overlap of G0_k itself with G_k: 0.936 [0.934, 0.938]. (nr) = metric not readable on this dataset.

Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | B0-list rho_bar | L3-mean rho_bar | L3-att rho_bar | L3-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a | n/a | n/a |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity within U_q; it never describes a deployable model, and no probe output, entry or halo row enters any retriever, feature, teacher or selection.
- A model that used these features without GNN outputs would be a competitor, and would need its own declaration under the QLS-U contract. This file does not open one.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).
- No later level or arm is opened by any reading here: every interpretation entry that names a next file needs its own declaration.

## Integrity, placement and compute

| dataset | queries | U_q rows | halo rows | entries | compile (min) | prototype entries compared | max diff / tolerance | compile meta sha256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| metaqa | 2,400 | 165,893 | 1,419,019 | 1,967,659 | 21 | 95,554,368 | 0.249 | `23fd1c11fd7dd099` |
| 2wiki | 6,290 | 245,055 | 527,882 | 1,580,750 | 9 | 141,151,680 | 0.249 | `337c140f081fa1a8` |
| squad | 5,841 | 176,880 | 239,982 | 1,905,184 | 3 | 101,882,880 | 0.249 | `53fddbe35596c3d4` |

Compile integrity (laptop, 6 threads; the served CRAG files exist only there): every pool size and every query's in-pool golds equal level 0's; x and its pool z-score equal level 0's stored values exactly at every U_q row, and each U_q row's halo row carries level 0's B0 to float16; the three family prototypes recomputed from the packed batch's edge lists equal the stored twin channels within 2 float16 ulps + 1e-6; the stored entries reproduce the packed rows whose target is a U_q row, with every in-edge count equal; q_tilde equals level 1's exactly. 0 mismatches.

| dataset | probes (min) | read (min) | device | driver | determinism warnings | mirror.json sha256 (host) | repeat |
|---|---:|---:|---|---|---:|---|---|
| metaqa | 20 | 0.6 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `049d5cd2c3719895` | bit-identical |
| 2wiki | 34 | 2.1 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `4f2db84a5a1b0d80` | n/a |
| squad | 24 | 1.7 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `de7c1917942862d2` | n/a |

Placement: `host_gpu_det` = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": false, "env": "mpr-cu128@62fc45e9e1ba"}, applied by placement_settings("det", 8) with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads, read back and checked by the equivalence file's environment_problems in every host process. Each host job ran from one commit; the LF sha256 of every repository module it imported was filed and checked against the committed files at the file stage. Deviations: GPU share --gpus 0.33 per job, not the declared 0.34 (three 0.34 shares exceed the one card in the rx scheduler); the three jobs shared the one RTX 4500 Ada; at the user's instruction all three jobs were queued at 18:29Z, before the 2wiki and squad inputs had landed, instead of each after its own transfer; a systems-only wrapper (outputs/mp_approx_l3/wait_landed.py, not committed, imports nothing from the repository) held each job until every file its mirror.json lists existed (metaqa 285 s, 2wiki 450 s, squad 600 s), then ran the committed python scripts/mp_approx_l3.py --stage run --dataset DS; the declared transfer order metaqa, 2wiki, squad was kept; rx run --mem 16 per job, not the gpu profile's 32 GB default (the scheduler had 114 of 119.7 GB reserved at submission); the laptop compiles ran before the code commit, from the bytes then committed: every compile meta.json records git_head 8029942 and script LF sha256 5e30b49f, which is b8c64b2's scripts/mp_approx_l3.py.
