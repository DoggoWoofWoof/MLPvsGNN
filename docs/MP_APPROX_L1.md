# MP-Approx level 1 (MP-ORACLE): fixed vector hop tokens, only their mixing learned

Declared in `configs/mp_approx_l1.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l1/record.json` (git-ignored), written 2026-09-29T14:10:07Z at `e0cb6d7`.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

This is level 1 of the MP-Approx ladder, on the MP-ORACLE track. Level 0 (`docs/MP_APPROX_L0.md`) read metaqa and 2wiki L0_LOW with its vector channels adding recovery, which licensed this file. As at level 0, the trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe or token is a retriever, a teacher or a feature.

## What was measured

- **Hop tokens**: for each family f in structural, ner, knn and FULL (the cell's own edge list, the three families concatenated), Z_{f,k} = P_f Z_{f,k-1} with Z_{f,0} = X, the served 1536-wide node embeddings. P_f is the mean over a node's in-neighbours within the pool; an edge listed twice counts twice, and there is no self-loop. Depths 1 to 3 are propagated over the whole pool in float64. Each token is the propagated vector times one fixed Gaussian JL matrix (1536 x 64, seed 20260929), so nothing about the propagation is learned.
- **Exact cosines** at 1536 wide: each token's cosine with the query, and each depth >= 1 token's cosine with the node's own embedding. A zero token (no in-neighbour) has cosine 0.
- **Bases** (no learned state passes between candidates): L1 = level 0's B0 (258) | cos_q (13) | cos_self (12) | tokens (13 x 64) | q_tilde * tokens (13 x 64), 1,947 columns, with q_tilde = qemb R; L1d1 = the same at depth <= 1, 907 columns. The ridge mixes the tokens linearly with weights bilinear in q_tilde; the MLP mixes them nonlinearly.
- **Probes**: level 0's ridge and 2x128 MLP, cross-fitted over level 0's five query folds with its inner split, seeds and thread count; every prediction is out-of-fold. Level 0's B0, B1 and B2 probes are **cited** from its pinned probe files, not refitted. Each cited rho_bar was recomputed here and matched level 0's filed value to within 1e-12.
- **Targets, recovery and U_q** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k), and the cell's neighbour message m_k_t. rho_M = (M(z(T_k) + rhat_k) - M(T_k)) / (M(G_k) - M(T_k)) for recall@5, full_coverage@5 and hit@1, read only where the gap's interval lies above 0. rho_bar is their mean. Metrics within U_q are an **upper bound** on full-pool metrics, so a LOW reading is robust and a HIGH one is not a full-pool claim.

Bands of rho_bar (level 0's thresholds): L1_HIGH (>= 0.75, interval low >= 0.50), L1_LOW (<= 0.25, interval high <= 0.50), L1_MID otherwise, NOT_READ with no readable metric. The dataset's reading is the band of L1-mlp on r_k.

## Readings

| dataset | queries | reading (L1-mlp on r) | rho_bar [95% CI] | level 0's B0-mlp | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|---|
| metaqa | 2,400 | **L1_LOW** | -0.935 [-1.088, -0.803] | -0.650 [-0.769, -0.549] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, L1d1-mlp); FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, B1-mlp) | l1_low_tokens_flat |
| 2wiki | 6,290 | **L1_LOW** | -0.202 [-0.267, -0.148] | 0.055 [-0.002, 0.109] | recall@5, full_coverage@5 | FIT_NOT_RANK (r, B1-mlp) | l1_low_tokens_flat |
| squad | 5,841 | **NOT_READ** | n/a | n/a | none | SEED_BOUND | message_tokens_add |

### Contrasts (paired bootstrap, the same resample matrix as level 0)

| dataset | hop_tokens (L1-mlp - B0-mlp) | beyond_twin_channels (L1-mlp - B1-mlp) | depth (L1-mlp - L1d1-mlp) | nonlinearity (L1-mlp - L1-ridge) | message_tokens (R2: L1-ridge - B1-ridge) | message_depth (R2: L1-ridge - L1d1-ridge) |
|---|---|---|---|---|---|---|
| metaqa | -0.285 [-0.370, -0.206] | -0.709 [-0.828, -0.617] | -0.068 [-0.132, -0.009] | 0.233 [0.144, 0.326] | -0.039 [-0.040, -0.037] | 0.011 [0.011, 0.012] |
| 2wiki | -0.257 [-0.305, -0.208] | -0.350 [-0.400, -0.300] | -0.107 [-0.145, -0.069] | 0.027 [-0.015, 0.068] | -0.036 [-0.037, -0.035] | 0.006 [0.006, 0.007] |
| squad | n/a | n/a | n/a | n/a | 0.004 [0.004, 0.005] | 0.008 [0.007, 0.008] |

What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):

- **l1_low_tokens_flat**: primary L1_LOW with the hop_tokens interval's lower end at or below 0: fixed propagation of node semantics adds no ranking effect that a probe recovers over the scalars. It points past levels 1 and 2 to query-conditioned neighbourhood weighting (level 3, attention-kernel moments) as the next file.
- **message_tokens_add**: the message_tokens interval above 0: the fixed tokens reconstruct more of the cell's neighbour message than the twin's learned depth-1 channels (the proposal's "current scalar compiler loses semantics")

## metaqa

2,400 V2_GATE queries (the declared subsample: 800 per hop of the 19,738), 165,893 U_q rows (mean |U_q| 69.1; mean pool 2017). Message edges per query, mean: structural 9,161, ner 274, knn 1,328.

Share of U_q rows whose token is zero (no in-neighbour at that depth): structural1 0.01, structural2 0.01, structural3 0.01, ner1 0.88, ner2 0.88, ner3 0.88, knn1 0.41, knn2 0.41, knn3 0.41, FULL1 0.00, FULL2 0.00, FULL3 0.00.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.061 | [0.055, 0.068] | yes |
| full_coverage@5 | 0.065 | [0.055, 0.074] | yes |
| hit@1 | 0.127 | [0.114, 0.140] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L1-mlp | 0.495 | 0.654 | 0.551 | 0.483 | 0.510 | -0.896 | -0.863 | -1.045 | -0.935 [-1.088, -0.803] | L1_LOW |
| L1d1-mlp | 0.518 | 0.665 | 0.552 | 0.479 | 0.515 | -0.861 | -0.799 | -0.940 | -0.867 [-1.005, -0.747] | L1_LOW |
| L1-ridge | 0.276 | 0.501 | 0.436 | 0.460 | 0.467 | -1.193 | -1.113 | -1.197 | -1.168 [-1.350, -1.017] | L1_LOW |
| L1d1-ridge | 0.268 | 0.490 | 0.425 | 0.454 | 0.465 | -1.168 | -1.051 | -1.245 | -1.155 [-1.325, -1.011] | L1_LOW |
| B0-ridge (cited) | 0.246 | 0.467 | 0.405 | 0.436 | 0.464 | -1.214 | -1.073 | -1.326 | -1.204 [-1.388, -1.049] | L1_LOW |
| B0-mlp (cited) | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L1_LOW |
| B1-ridge (cited) | 0.319 | 0.533 | 0.437 | 0.453 | 0.493 | -0.858 | -0.765 | -0.913 | -0.845 [-0.988, -0.722] | L1_LOW |
| B1-mlp (cited) | 0.663 | 0.759 | 0.630 | 0.514 | 0.576 | -0.214 | -0.214 | -0.251 | -0.226 [-0.305, -0.158] | L1_LOW |
| B2-ridge (cited) | 0.373 | 0.512 | 0.323 | 0.247 | 0.524 | -0.100 | -0.103 | -0.208 | -0.137 [-0.190, -0.094] | L1_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L1_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L1_HIGH |

Top-5 overlap of T_k itself with G_k: 0.529 [0.520, 0.538]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L1-ridge | 0.633 | 0.767 | 0.659 | 0.808 | 0.418 | 0.529 | 0.455 | 0.414 | 0.466 [0.449, 0.483] | L1_MID |
| L1d1-ridge | 0.628 | 0.764 | 0.655 | 0.802 | 0.414 | 0.517 | 0.445 | 0.407 | 0.457 [0.439, 0.474] | L1_MID |
| B0-ridge (cited) | 0.610 | 0.754 | 0.640 | 0.788 | 0.407 | 0.498 | 0.431 | 0.379 | 0.436 [0.418, 0.453] | L1_MID |
| B1-ridge (cited) | 0.674 | 0.782 | 0.673 | 0.832 | 0.455 | 0.614 | 0.536 | 0.485 | 0.545 [0.528, 0.561] | L1_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L1_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L1_HIGH |

Top-5 overlap of G0_k itself with G_k: 0.246 [0.237, 0.254]. (nr) = metric not readable on this dataset.

Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Neighbour messages (m_k_t)

| probe | R2 step 1 | R2 step 2 | R2 step 3 | R2 step mean [95% CI] | cosine step 1 | cosine step 2 | cosine step 3 |
|---|---:|---:|---:|---|---:|---:|---:|
| L1-ridge | 0.548 | 0.469 | 0.414 | 0.477 [0.472, 0.482] | 0.720 | 0.664 | 0.619 |
| L1d1-ridge | 0.537 | 0.457 | 0.403 | 0.466 [0.461, 0.471] | 0.714 | 0.657 | 0.612 |
| B0-ridge (cited) | 0.502 | 0.430 | 0.381 | 0.438 [0.433, 0.443] | 0.696 | 0.639 | 0.594 |
| B1-ridge (cited) | 0.577 | 0.514 | 0.457 | 0.516 [0.511, 0.520] | 0.740 | 0.689 | 0.646 |

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | L1-mlp rho_bar | L1d1-mlp rho_bar | B0-mlp rho_bar | B1-mlp rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.241 [-2.283, -0.769] | -0.943 [-1.727, -0.509] | -1.039 [-1.842, -0.582] | 0.099 [-0.201, 0.320] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -2.001 [-2.680, -1.551] | -1.854 [-2.474, -1.435] | -1.049 [-1.466, -0.759] | -0.270 [-0.494, -0.104] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.594 [-0.731, -0.487] | -0.579 [-0.710, -0.470] | -0.489 [-0.602, -0.386] | -0.255 [-0.345, -0.177] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.755 [-0.960, -0.595] | -0.675 [-0.863, -0.530] | -0.619 [-0.798, -0.474] | -0.120 [-0.228, -0.033] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -1.084 [-1.384, -0.862] | -0.969 [-1.227, -0.779] | -0.577 [-0.797, -0.418] | -0.168 [-0.292, -0.064] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -1.109 [-1.542, -0.799] | -1.186 [-1.635, -0.853] | -0.898 [-1.256, -0.631] | -0.656 [-0.965, -0.417] | 1.013 [0.971, 1.057] |

## 2wiki

6,290 V2_GATE queries (all of them), 245,055 U_q rows (mean |U_q| 39.0; mean pool 105). Message edges per query, mean: structural 267, ner 157, knn 68.

Share of U_q rows whose token is zero (no in-neighbour at that depth): structural1 0.19, structural2 0.19, structural3 0.19, ner1 0.51, ner2 0.51, ner3 0.51, knn1 0.51, knn2 0.51, knn3 0.51, FULL1 0.13, FULL2 0.13, FULL3 0.13.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.048 | [0.045, 0.051] | yes |
| full_coverage@5 | 0.104 | [0.097, 0.111] | yes |
| hit@1 | -0.007 | [-0.012, -0.002] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L1-mlp | 0.369 | 0.583 | 0.459 | 0.451 | 0.740 | -0.161 | -0.243 | -1.667 (nr) | -0.202 [-0.267, -0.148] | L1_LOW |
| L1d1-mlp | 0.406 | 0.610 | 0.475 | 0.472 | 0.748 | -0.058 | -0.132 | -1.227 (nr) | -0.095 [-0.156, -0.040] | L1_LOW |
| L1-ridge | 0.335 | 0.555 | 0.410 | 0.422 | 0.737 | -0.196 | -0.262 | -1.932 (nr) | -0.229 [-0.289, -0.168] | L1_LOW |
| L1d1-ridge | 0.330 | 0.552 | 0.403 | 0.416 | 0.737 | -0.218 | -0.283 | -2.106 (nr) | -0.251 [-0.313, -0.190] | L1_LOW |
| B0-ridge (cited) | 0.312 | 0.534 | 0.382 | 0.402 | 0.734 | -0.201 | -0.257 | -1.826 (nr) | -0.229 [-0.292, -0.170] | L1_LOW |
| B0-mlp (cited) | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L1_LOW |
| B1-ridge (cited) | 0.390 | 0.614 | 0.473 | 0.460 | 0.741 | -0.265 | -0.323 | -2.189 (nr) | -0.294 [-0.362, -0.224] | L1_LOW |
| B1-mlp (cited) | 0.597 | 0.753 | 0.627 | 0.543 | 0.780 | 0.193 | 0.103 | -0.288 (nr) | 0.148 [0.096, 0.195] | L1_LOW |
| B2-ridge (cited) | 0.373 | 0.601 | 0.404 | 0.305 | 0.731 | -0.054 | -0.097 | -0.295 (nr) | -0.076 [-0.122, -0.030] | L1_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L1_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L1_HIGH |

Top-5 overlap of T_k itself with G_k: 0.704 [0.700, 0.707]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L1-ridge | 0.464 | 0.621 | 0.562 | 0.589 | 0.714 | 0.345 | 0.312 | 3.156 (nr) | 0.329 [0.314, 0.344] | L1_MID |
| L1d1-ridge | 0.456 | 0.619 | 0.555 | 0.580 | 0.711 | 0.334 | 0.302 | 3.099 (nr) | 0.318 [0.303, 0.333] | L1_MID |
| B0-ridge (cited) | 0.423 | 0.603 | 0.529 | 0.540 | 0.702 | 0.282 | 0.267 | 3.109 (nr) | 0.275 [0.260, 0.290] | L1_MID |
| B1-ridge (cited) | 0.515 | 0.636 | 0.586 | 0.646 | 0.731 | 0.441 | 0.390 | 3.019 (nr) | 0.416 [0.401, 0.431] | L1_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L1_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L1_HIGH |

Top-5 overlap of G0_k itself with G_k: 0.652 [0.647, 0.657]. (nr) = metric not readable on this dataset.

Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Neighbour messages (m_k_t)

| probe | R2 step 1 | R2 step 2 | R2 step 3 | R2 step mean [95% CI] | cosine step 1 | cosine step 2 | cosine step 3 |
|---|---:|---:|---:|---|---:|---:|---:|
| L1-ridge | 0.511 | 0.471 | 0.464 | 0.482 [0.479, 0.485] | 0.718 | 0.689 | 0.677 |
| L1d1-ridge | 0.504 | 0.465 | 0.459 | 0.476 [0.473, 0.478] | 0.716 | 0.687 | 0.675 |
| B0-ridge (cited) | 0.478 | 0.441 | 0.435 | 0.451 [0.448, 0.454] | 0.705 | 0.673 | 0.661 |
| B1-ridge (cited) | 0.548 | 0.505 | 0.499 | 0.518 [0.515, 0.520] | 0.733 | 0.706 | 0.696 |

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | L1-mlp rho_bar | L1d1-mlp rho_bar | B0-mlp rho_bar | B1-mlp rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | -0.115 [-0.192, -0.046] | 0.009 [-0.063, 0.073] | 0.193 [0.128, 0.255] | 0.207 [0.143, 0.269] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.387 [-0.501, -0.290] | -0.314 [-0.425, -0.221] | -0.232 [-0.343, -0.141] | 0.026 [-0.064, 0.102] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.025 [-1.155, 0.592] | 0.290 [-0.478, 0.868] | 0.567 [-0.025, 1.209] | 0.608 [-0.000, 1.221] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | -0.221 [-0.290, -0.164] | -0.118 [-0.182, -0.063] | 0.033 [-0.024, 0.087] | 0.120 [0.069, 0.168] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.258 [-0.001, 0.481] | 0.437 [0.183, 0.692] | 0.476 [0.267, 0.718] | 0.575 [0.373, 0.819] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a | n/a |

## squad

5,841 V2_GATE queries (all of them; the control), 176,880 U_q rows (mean |U_q| 30.3; mean pool 50). Message edges per query, mean: structural 251, ner 112, knn 83.

Share of U_q rows whose token is zero (no in-neighbour at that depth): structural1 0.56, structural2 0.56, structural3 0.56, ner1 0.48, ner2 0.48, ner3 0.48, knn1 0.26, knn2 0.26, knn3 0.26, FULL1 0.12, FULL2 0.12, FULL3 0.12.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.004 | [-0.007, -0.002] | no |
| full_coverage@5 | -0.004 | [-0.007, -0.002] | no |
| hit@1 | -0.009 | [-0.013, -0.005] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L1-mlp | 0.492 | 0.655 | 0.510 | 0.450 | 0.810 | -0.224 (nr) | -0.224 (nr) | 0.138 (nr) | n/a | NOT_READ |
| L1d1-mlp | 0.514 | 0.670 | 0.523 | 0.460 | 0.814 | -0.276 (nr) | -0.276 (nr) | 0.182 (nr) | n/a | NOT_READ |
| L1-ridge | 0.401 | 0.576 | 0.402 | 0.355 | 0.791 | -0.382 (nr) | -0.382 (nr) | 0.025 (nr) | n/a | NOT_READ |
| L1d1-ridge | 0.395 | 0.568 | 0.390 | 0.336 | 0.789 | -0.289 (nr) | -0.289 (nr) | -0.019 (nr) | n/a | NOT_READ |
| B0-ridge (cited) | 0.363 | 0.536 | 0.355 | 0.311 | 0.783 | -0.408 (nr) | -0.408 (nr) | -0.050 (nr) | n/a | NOT_READ |
| B0-mlp (cited) | 0.443 | 0.600 | 0.421 | 0.348 | 0.793 | -0.342 (nr) | -0.342 (nr) | -0.252 (nr) | n/a | NOT_READ |
| B1-ridge (cited) | 0.543 | 0.698 | 0.528 | 0.443 | 0.807 | -0.947 (nr) | -0.947 (nr) | 0.138 (nr) | n/a | NOT_READ |
| B1-mlp (cited) | 0.710 | 0.813 | 0.681 | 0.572 | 0.846 | -0.224 (nr) | -0.224 (nr) | 0.233 (nr) | n/a | NOT_READ |
| B2-ridge (cited) | 0.447 | 0.622 | 0.416 | 0.272 | 0.786 | -1.066 (nr) | -1.066 (nr) | -0.390 (nr) | n/a | NOT_READ |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.746 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | -0.245 | 0.167 | 0.278 | 0.316 | 0.724 | 2.105 (nr) | 2.105 (nr) | 1.836 (nr) | n/a | NOT_READ |

Top-5 overlap of T_k itself with G_k: 0.746 [0.743, 0.749]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.001 | [-0.002, 0.001] | no |
| full_coverage@5 | -0.001 | [-0.002, 0.001] | no |
| hit@1 | -0.001 | [-0.002, 0.000] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L1-ridge | 0.281 | 0.543 | 0.460 | 0.392 | 0.945 | 1.000 (nr) | 1.000 (nr) | 1.667 (nr) | n/a | NOT_READ |
| L1d1-ridge | 0.271 | 0.536 | 0.446 | 0.371 | 0.945 | 0.889 (nr) | 0.889 (nr) | 1.476 (nr) | n/a | NOT_READ |
| B0-ridge (cited) | 0.251 | 0.523 | 0.424 | 0.377 | 0.944 | 1.333 (nr) | 1.333 (nr) | 1.762 (nr) | n/a | NOT_READ |
| B1-ridge (cited) | 0.283 | 0.550 | 0.458 | 0.402 | 0.945 | 0.667 (nr) | 0.667 (nr) | 1.810 (nr) | n/a | NOT_READ |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Top-5 overlap of G0_k itself with G_k: 0.936 [0.934, 0.938]. (nr) = metric not readable on this dataset.

Seed reproducibility of the targets (within-query-centred correlation across GNN seeds, mean of the three pairs): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Neighbour messages (m_k_t)

| probe | R2 step 1 | R2 step 2 | R2 step 3 | R2 step mean [95% CI] | cosine step 1 | cosine step 2 | cosine step 3 |
|---|---:|---:|---:|---|---:|---:|---:|
| L1-ridge | 0.716 | 0.690 | 0.669 | 0.692 [0.689, 0.694] | 0.787 | 0.777 | 0.769 |
| L1d1-ridge | 0.708 | 0.682 | 0.661 | 0.684 [0.681, 0.687] | 0.780 | 0.770 | 0.762 |
| B0-ridge (cited) | 0.694 | 0.668 | 0.647 | 0.670 [0.667, 0.672] | 0.770 | 0.759 | 0.751 |
| B1-ridge (cited) | 0.712 | 0.685 | 0.665 | 0.687 [0.684, 0.690] | 0.781 | 0.770 | 0.762 |

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | L1-mlp rho_bar | L1d1-mlp rho_bar | B0-mlp rho_bar | B1-mlp rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a | n/a |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity within U_q. It never describes a deployable model, and no probe output or token enters any retriever, feature, teacher or selection.
- A model that used these tokens without GNN outputs would be a competitor, and would need its own declaration under the QLS-U contract. This file does not open one.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).
- Level 2 is not opened by any reading here: every interpretation entry names a next file, and each needs its own declaration.

## Integrity and compute

| dataset | queries | U_q rows | token pass (min) | probes (min) | prototype entries compared | max abs diff | max diff / tolerance | tokens meta sha256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| metaqa | 2,400 | 165,893 | 7 | 24 | 95,554,368 | 1.22e-04 | 0.249 | `56e7b9161514d2e5` |
| 2wiki | 6,290 | 245,055 | 5 | 28 | 141,151,680 | 1.22e-04 | 0.249 | `efc7f0bc01e05cad` |
| squad | 5,841 | 176,880 | 2 | 17 | 101,882,880 | 7.05e-05 | 0.249 | `2503b30da07e9c99` |

Every pool size and every query's in-pool golds equal level 0's (integrity.rows). At every U_q row the three family prototypes, recomputed from this pass's edge lists and embeddings with each twin seed's own projections, equal the channels level 0 stored as the twin scored the query, within 2 float16 ulps + 1e-6 (integrity.edges). This ties each family's edge list to what the models read. The cited probes' rho_bar match level 0's filed values (largest |diff| metaqa 0.0e+00, 2wiki 0.0e+00, squad 0.0e+00).

Placement: laptop CPU for every stage (placement in the declaration). Tokens at 6 threads in one lane, probes at 4 threads per process. The host GPU is barred (configs/cpu_gpu_equivalence.yaml: NOT_EQUIVALENT), and the upload the host CPU would need does not finish sooner.
