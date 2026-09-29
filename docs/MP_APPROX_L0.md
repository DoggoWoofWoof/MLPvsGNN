# MP-Approx level 0 (MP-ORACLE): how much of the GNN's effect node-local compiled information predicts

Declared in `configs/mp_approx_l0.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l0/record.json` (git-ignored), written 2026-09-29T10:39:24Z at `5434594`.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

This is level 0 of the MP-Approx ladder, on the MP-ORACLE track. The trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe is a retriever, a teacher or a feature.

## What was measured

- **Scored functions** (pilot trio checkpoints, V2_GATE rows only): the twin `u_mlp_v2_mix__H128__s{k}` (T_k), the GNN `u_gnn_v2_ef__H128__s{k}` (G_k) and the GNN's no-edge counterfactual (G0_k: one forward with `message_passing = False`), k = 0, 1, 2.
- **Targets**, centred within the query: r_k = z(G_k) - z(T_k) (primary; the GNN's residual over its matched twin), e_k = z(G_k) - z(G0_k) (what the cell's edges add inside the trained GNN), and m_k_t, the cell's neighbour message at step t (128-d).
- **Bases** (node-local, nothing propagated): B0 = the 129 contract scalars (signed log1p plus within-query z; 258 columns); B1 = B0 plus the twin's 515 vector channels (semantic head, semantic_difference, the three hop-1 family prototypes, seed reach); B2 = the twin's own body output and base_z.
- **Probes**: ridge and a 2x128 MLP, cross-fitted over five query folds (D0.2's rule); every prediction is out-of-fold.
- **Recovery** rho_M = (M(z(T_k) + rhat_k) - M(T_k)) / (M(G_k) - M(T_k)), means over seeds and queries, for M in recall@5, full_coverage@5 and hit@1; read only where the GNN-twin gap's 95% interval lies above 0. rho_bar is their mean. rho is 0 for the twin itself and 1 for an approximation that ranks U_q as G_k does; it can exceed 1 (the approximation ranks the golds better than G_k within U_q) or fall below 0.
- For e_k every measure takes G0_k in T_k's place: the gap is M(G_k) - M(G0_k), and DPR counts the pairs G0_k and G_k order oppositely.
- **U_q**: the in-pool golds plus the top 20 of each of the nine functions. Every endpoint metric is exact within U_q (checked on every query); an approximation's metrics within U_q are an **upper bound** on its full-pool metrics, so a LOW reading is robust and a HIGH one is not a full-pool claim.

Bands of rho_bar: L0_HIGH (>= 0.75, interval low >= 0.50), L0_LOW (<= 0.25, interval high <= 0.50), L0_MID otherwise, NOT_READ with no readable metric. The dataset's reading is the band of B0-mlp on r_k.

## Readings

| dataset | queries | reading (B0-mlp on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|
| metaqa | 2,400 | **L0_LOW** | -0.650 [-0.769, -0.549] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, B1-mlp); MESSAGE_NOT_RETRIEVAL | primary_low_vectors_add, b1_low, message_not_retrieval |
| 2wiki | 6,290 | **L0_LOW** | 0.055 [-0.002, 0.109] | recall@5, full_coverage@5 | FIT_NOT_RANK (r, B1-mlp); MESSAGE_NOT_RETRIEVAL | primary_low_vectors_add, b1_low, message_not_retrieval |
| squad | 5,841 | **NOT_READ** | n/a | none | SEED_BOUND | none |

What the interpretation entries license, as filed before any number:

- **b1_low**: B1-mlp L0_LOW: node-local information, depth-1 vectors included, does not predict the effect. What the cell adds lies in depth >= 2, query-conditioned weighting or typed composition, the ground of levels 1 to 4.
- **message_not_retrieval**: the proposal's "scorer/objective problem" diagnosis
- **primary_low_vectors_add**: primary L0_LOW with the vector_channels interval above 0: the twin's depth-1 vector channels carry part of what the scalars lose, which is the proposal's "scalar compiler loses semantics" diagnosis at depth 1. It licenses level 1 (depth >= 2 vector tokens) as the next file.

## metaqa

2,400 V2_GATE queries (the declared subsample: 800 per hop of the 19,738), 165,893 U_q rows (mean |U_q| 69.1; mean pool 2017).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

Denominator: the gap M(G_k) - M(T_k), seed mean, with its 95% interval.

| metric | gap | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.061 | [0.055, 0.068] | yes |
| full_coverage@5 | 0.065 | [0.055, 0.074] | yes |
| hit@1 | 0.127 | [0.114, 0.140] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-ridge | 0.246 | 0.467 | 0.405 | 0.436 | 0.464 | -1.214 | -1.073 | -1.326 | -1.204 [-1.388, -1.049] | L0_LOW |
| B0-mlp | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L0_LOW |
| B1-ridge | 0.319 | 0.533 | 0.437 | 0.453 | 0.493 | -0.858 | -0.765 | -0.913 | -0.845 [-0.988, -0.722] | L0_LOW |
| B1-mlp | 0.663 | 0.759 | 0.630 | 0.514 | 0.576 | -0.214 | -0.214 | -0.251 | -0.226 [-0.305, -0.158] | L0_LOW |
| B2-ridge | 0.373 | 0.512 | 0.323 | 0.247 | 0.524 | -0.100 | -0.103 | -0.208 | -0.137 [-0.190, -0.094] | L0_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L0_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L0_HIGH |

Top-5 overlap of T_k itself with G_k: 0.529 [0.520, 0.538]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)

Denominator: the gap M(G_k) - M(G0_k), seed mean, with its 95% interval.

| metric | gap | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-ridge | 0.610 | 0.754 | 0.640 | 0.788 | 0.407 | 0.498 | 0.431 | 0.379 | 0.436 [0.418, 0.453] | L0_MID |
| B1-ridge | 0.674 | 0.782 | 0.673 | 0.832 | 0.455 | 0.614 | 0.536 | 0.485 | 0.545 [0.528, 0.561] | L0_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L0_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L0_HIGH |

Top-5 overlap of G0_k itself with G_k: 0.246 [0.237, 0.254]. (nr) = metric not readable on this dataset.

### Seed reproducibility of the targets

Within-query-centred correlation across GNN seeds (mean of pairs 0-1, 0-2, 1-2): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Neighbour messages (m_k_t)

| probe | R2 step 1 | R2 step 2 | R2 step 3 | cosine step 1 | cosine step 2 | cosine step 3 |
|---|---:|---:|---:|---:|---:|---:|
| B0-ridge | 0.502 | 0.430 | 0.381 | 0.696 | 0.639 | 0.594 |
| B1-ridge | 0.577 | 0.514 | 0.457 | 0.740 | 0.689 | 0.646 |

### Contrasts of rho_bar (paired bootstrap)

- vector_channels: rho_bar(B1-mlp) - rho_bar(B0-mlp) = 0.424 [0.350, 0.514]
- nonlinearity: rho_bar(B0-mlp) - rho_bar(B0-ridge) = 0.554 [0.449, 0.677]
- twin_representation: rho_bar(B2-ridge) - rho_bar(B1-mlp) = 0.089 [0.019, 0.161]

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | B1-mlp rho_bar | B2-ridge rho_bar | other-seed rho_bar |
|---|---:|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.039 [-1.842, -0.582] | 0.099 [-0.201, 0.320] | -0.647 [-1.062, -0.397] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -1.049 [-1.466, -0.759] | -0.270 [-0.494, -0.104] | -0.175 [-0.327, -0.057] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.489 [-0.602, -0.386] | -0.255 [-0.345, -0.177] | -0.059 [-0.108, -0.017] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.619 [-0.798, -0.474] | -0.120 [-0.228, -0.033] | -0.178 [-0.262, -0.111] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -0.577 [-0.797, -0.418] | -0.168 [-0.292, -0.064] | -0.093 [-0.173, -0.024] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -0.898 [-1.256, -0.631] | -0.656 [-0.965, -0.417] | -0.128 [-0.245, -0.034] | 1.013 [0.971, 1.057] |

## 2wiki

6,290 V2_GATE queries (all of them), 245,055 U_q rows (mean |U_q| 39.0; mean pool 105).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

Denominator: the gap M(G_k) - M(T_k), seed mean, with its 95% interval.

| metric | gap | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.048 | [0.045, 0.051] | yes |
| full_coverage@5 | 0.104 | [0.097, 0.111] | yes |
| hit@1 | -0.007 | [-0.012, -0.002] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-ridge | 0.312 | 0.534 | 0.382 | 0.402 | 0.734 | -0.201 | -0.257 | -1.826 (nr) | -0.229 [-0.292, -0.170] | L0_LOW |
| B0-mlp | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L0_LOW |
| B1-ridge | 0.390 | 0.614 | 0.473 | 0.460 | 0.741 | -0.265 | -0.323 | -2.189 (nr) | -0.294 [-0.362, -0.224] | L0_LOW |
| B1-mlp | 0.597 | 0.753 | 0.627 | 0.543 | 0.780 | 0.193 | 0.103 | -0.288 (nr) | 0.148 [0.096, 0.195] | L0_LOW |
| B2-ridge | 0.373 | 0.601 | 0.404 | 0.305 | 0.731 | -0.054 | -0.097 | -0.295 (nr) | -0.076 [-0.122, -0.030] | L0_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L0_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L0_HIGH |

Top-5 overlap of T_k itself with G_k: 0.704 [0.700, 0.707]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)

Denominator: the gap M(G_k) - M(G0_k), seed mean, with its 95% interval.

| metric | gap | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-ridge | 0.423 | 0.603 | 0.529 | 0.540 | 0.702 | 0.282 | 0.267 | 3.109 (nr) | 0.275 [0.260, 0.290] | L0_MID |
| B1-ridge | 0.515 | 0.636 | 0.586 | 0.646 | 0.731 | 0.441 | 0.390 | 3.019 (nr) | 0.416 [0.401, 0.431] | L0_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L0_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L0_HIGH |

Top-5 overlap of G0_k itself with G_k: 0.652 [0.647, 0.657]. (nr) = metric not readable on this dataset.

### Seed reproducibility of the targets

Within-query-centred correlation across GNN seeds (mean of pairs 0-1, 0-2, 1-2): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Neighbour messages (m_k_t)

| probe | R2 step 1 | R2 step 2 | R2 step 3 | cosine step 1 | cosine step 2 | cosine step 3 |
|---|---:|---:|---:|---:|---:|---:|
| B0-ridge | 0.478 | 0.441 | 0.435 | 0.705 | 0.673 | 0.661 |
| B1-ridge | 0.548 | 0.505 | 0.499 | 0.733 | 0.706 | 0.696 |

### Contrasts of rho_bar (paired bootstrap)

- vector_channels: rho_bar(B1-mlp) - rho_bar(B0-mlp) = 0.093 [0.048, 0.139]
- nonlinearity: rho_bar(B0-mlp) - rho_bar(B0-ridge) = 0.284 [0.239, 0.329]
- twin_representation: rho_bar(B2-ridge) - rho_bar(B1-mlp) = -0.224 [-0.270, -0.176]

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | B1-mlp rho_bar | B2-ridge rho_bar | other-seed rho_bar |
|---|---:|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | 0.193 [0.128, 0.255] | 0.207 [0.143, 0.269] | -0.014 [-0.067, 0.041] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.232 [-0.343, -0.141] | 0.026 [-0.064, 0.102] | -0.209 [-0.285, -0.137] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.567 [-0.025, 1.209] | 0.608 [-0.000, 1.221] | 0.312 [-0.440, 1.000] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | 0.033 [-0.024, 0.087] | 0.120 [0.069, 0.168] | -0.099 [-0.145, -0.055] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.476 [0.267, 0.718] | 0.575 [0.373, 0.819] | 0.372 [0.160, 0.651] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a |

## squad

5,841 V2_GATE queries (all of them; the control), 176,880 U_q rows (mean |U_q| 30.3; mean pool 50).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

Denominator: the gap M(G_k) - M(T_k), seed mean, with its 95% interval.

| metric | gap | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.004 | [-0.007, -0.002] | no |
| full_coverage@5 | -0.004 | [-0.007, -0.002] | no |
| hit@1 | -0.009 | [-0.013, -0.005] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-ridge | 0.363 | 0.536 | 0.355 | 0.311 | 0.783 | -0.408 (nr) | -0.408 (nr) | -0.050 (nr) | n/a | NOT_READ |
| B0-mlp | 0.443 | 0.600 | 0.421 | 0.348 | 0.793 | -0.342 (nr) | -0.342 (nr) | -0.252 (nr) | n/a | NOT_READ |
| B1-ridge | 0.543 | 0.698 | 0.528 | 0.443 | 0.807 | -0.947 (nr) | -0.947 (nr) | 0.138 (nr) | n/a | NOT_READ |
| B1-mlp | 0.710 | 0.813 | 0.681 | 0.572 | 0.846 | -0.224 (nr) | -0.224 (nr) | 0.233 (nr) | n/a | NOT_READ |
| B2-ridge | 0.447 | 0.622 | 0.416 | 0.272 | 0.786 | -1.066 (nr) | -1.066 (nr) | -0.390 (nr) | n/a | NOT_READ |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.746 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | -0.245 | 0.167 | 0.278 | 0.316 | 0.724 | 2.105 (nr) | 2.105 (nr) | 1.836 (nr) | n/a | NOT_READ |

Top-5 overlap of T_k itself with G_k: 0.746 [0.743, 0.749]. (nr) = metric not readable on this dataset.

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)

Denominator: the gap M(G_k) - M(G0_k), seed mean, with its 95% interval.

| metric | gap | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.001 | [-0.002, 0.001] | no |
| full_coverage@5 | -0.001 | [-0.002, 0.001] | no |
| hit@1 | -0.001 | [-0.002, 0.000] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-ridge | 0.251 | 0.523 | 0.424 | 0.377 | 0.944 | 1.333 (nr) | 1.333 (nr) | 1.762 (nr) | n/a | NOT_READ |
| B1-ridge | 0.283 | 0.550 | 0.458 | 0.402 | 0.945 | 0.667 (nr) | 0.667 (nr) | 1.810 (nr) | n/a | NOT_READ |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Top-5 overlap of G0_k itself with G_k: 0.936 [0.934, 0.938]. (nr) = metric not readable on this dataset.

### Seed reproducibility of the targets

Within-query-centred correlation across GNN seeds (mean of pairs 0-1, 0-2, 1-2): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Neighbour messages (m_k_t)

| probe | R2 step 1 | R2 step 2 | R2 step 3 | cosine step 1 | cosine step 2 | cosine step 3 |
|---|---:|---:|---:|---:|---:|---:|
| B0-ridge | 0.694 | 0.668 | 0.647 | 0.770 | 0.759 | 0.751 |
| B1-ridge | 0.712 | 0.685 | 0.665 | 0.781 | 0.770 | 0.762 |

### Contrasts of rho_bar (paired bootstrap)

- vector_channels: rho_bar(B1-mlp) - rho_bar(B0-mlp) = n/a (no readable metric)
- nonlinearity: rho_bar(B0-mlp) - rho_bar(B0-ridge) = n/a (no readable metric)
- twin_representation: rho_bar(B2-ridge) - rho_bar(B1-mlp) = n/a (no readable metric)

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | B1-mlp rho_bar | B2-ridge rho_bar | other-seed rho_bar |
|---|---:|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery is a statement of capacity within U_q, never of a deployable model, and no probe output enters any retriever, feature, teacher or selection.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).
- Level 1 is not opened by any reading here; v2.1 and v2.2A stay closed as mechanisms and v2.2B is not opened.

## Integrity and compute

| dataset | queries | U_q rows | chunk (queries) | scoring threads | scoring (min) | probes (min) | stored-metric, message-split and U_q checks | sidecar meta sha256 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| metaqa | 2,400 | 165,893 | 11 | 6 | 70 | 27 | 0 mismatches | `d8d618cf128327c8` |
| 2wiki | 6,290 | 245,055 | 227 | 6 | 13 | 31 | 0 mismatches | `78225a25ce4a803a` |
| squad | 5,841 | 176,880 | 479 | 6 | 8 | 24 | 0 mismatches | `b520ccfb79cb02ec` |

Placement: laptop CPU for every stage (placement in the declaration). Scoring at 6 threads (the stored pilot records' count), probes at 4 threads per process.
