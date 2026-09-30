# MP-Approx level 4 (MP-ORACLE): compiled typed-walk and typed-subtree sketches beside one-hop attention

Declared in `configs/mp_approx_l4.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l4/record.json` (git-ignored), assembled 2026-09-30T00:07:44Z at `0a21dfe` from the host's per-dataset `read.json` files, without arithmetic.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

Level 4 of the MP-Approx ladder, on the MP-ORACLE track, declared with level 5 (neighbour moments) on the user's message of 2026-09-30. It gives each candidate a fixed, compiled summary of the typed walks that reach it, from the query's seeds and from anywhere in the pool graph the GNN's cell reads, and asks what that adds to level 3's learned one-hop attention over fixed rows. As at every earlier level, the trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe or sketch is a retriever, a teacher or a feature.

**Placement: host-native for every number.** The sketches were compiled on the laptop CPU (the served CRAG pool graph exists only there), checked query by query against level 3's stored entries, and pushed with a mirror manifest. Every probe this file compares, B0-mlp and L3-att included, was fitted and read on host_gpu_det as a new draw. No number here is set beside a number made on the laptop.

## What was measured

- **Graph**: the FULL message edges u -> v of the query's pool (structural, ner and knn, in packed order, duplicates kept), built by the frozen `pack_queries_v2` and checked against level 3's stored one-hop entries for every query (integrity.graph).
- **Edge token**: a 64-bit splitmix64 chain over (family, dir_fwd > 0.5, dir_bwd > 0.5, the four relation slots as dataset-local ids sorted ascending). Relation ids are hashed; the probe sees hash buckets only, never a relation-id parameter.
- **Sketches** (267 columns per U_q row, float16): a tensor count sketch of the typed walks of length 1, 2 and 3 from the query's seeds (3 x 64 buckets; bucket = sum of position hashes mod 64, sign = product of position signs, so r1 -> r2 and r2 -> r1 are different sequences); the same over every typed walk of length 1 and 2 that ends at the node from anywhere in the pool, the linear WL surrogate (2 x 32); log1p of the exact walk totals (5); and the path-query match, each walk's mean rel_compat averaged over and maximised over the seed walks of length 1, 2, 3 (6). The counts are integers held exactly in float64, computed by a dynamic programme over the edges, never by enumeration; a laptop test holds them equal to a brute-force enumeration of the walks.
- **L4-mlp**: level 0's MLP on [b_v | p_v], no attention. **L4-att**: level 3's L3-att with the readout widened to [b_v | p_v | m_v]; its scores, values and heads are level 3's and initialise as level 3's for the same seed. **L4-list**: L4-att under level 3's listwise objective.
- **Refitted from level 3** with its code, unchanged: B0-mlp (node-local) and L3-att (learned query-conditioned one-hop attention).
- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** on full-pool metrics. L4_HIGH (>= 0.75, interval low >= 0.50), L4_LOW (<= 0.25, interval high <= 0.50), L4_MID otherwise.

## Readings

| dataset | queries | reading (L4-att on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|
| metaqa | 2,400 | **L4_LOW** | -0.182 [-0.278, -0.097] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, L4-mlp); FIT_NOT_RANK (r, L3-att); FIT_NOT_RANK (r, L4-att) | l4_paths_flat, paths_node_local_adds, objective_adds, edge_effect_paths |
| 2wiki | 6,290 | **L4_HIGH** | 0.819 [0.774, 0.859] | recall@5, full_coverage@5 | none | l4_high, l4_paths_flat, paths_node_local_adds, objective_adds, edge_effect_paths |
| squad | 5,841 | **NOT_READ** | n/a | none | SEED_BOUND | none |

### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)

| dataset | B0-mlp | L4-mlp | L3-att | L4-att | L4-list | ref:other_seed |
|---|---|---|---|---|---|---|
| metaqa | -0.650 [-0.769, -0.549] (L4_LOW) | -0.585 [-0.711, -0.477] (L4_LOW) | -0.076 [-0.161, 0.002] (L4_LOW) | -0.182 [-0.278, -0.097] (L4_LOW) | 0.222 [0.153, 0.289] (L4_LOW) | 1.064 [1.044, 1.086] (L4_HIGH) |
| 2wiki | 0.055 [-0.002, 0.109] (L4_LOW) | 0.254 [0.203, 0.303] (L4_MID) | 0.799 [0.759, 0.838] (L4_HIGH) | 0.819 [0.774, 0.859] (L4_HIGH) | 0.891 [0.851, 0.928] (L4_HIGH) | 0.867 [0.837, 0.893] (L4_HIGH) |
| squad | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) |

### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)

| contrast | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| paths_over_attention | L4-att - L3-att, r | -0.106 [-0.154, -0.058] | 0.021 [-0.004, 0.047] | n/a |
| paths_over_node_local | L4-mlp - B0-mlp, r | 0.065 [0.004, 0.130] | 0.199 [0.163, 0.238] | n/a |
| paths_vs_attention | L4-mlp - L3-att, r | -0.509 [-0.596, -0.435] | -0.545 [-0.597, -0.493] | n/a |
| objective_paths | L4-list - L4-att, r | 0.404 [0.340, 0.475] | 0.072 [0.042, 0.099] | n/a |
| edge_paths_over_attention | L4-att - L3-att, e | 0.011 [0.003, 0.020] | 0.007 [-0.000, 0.014] | n/a |

What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):

- **edge_effect_paths**: the band of L4-att on e is L4_HIGH, or edge_paths_over_attention's interval lies above 0
- **l4_high**: primary L4_HIGH: one-hop attention over fixed rows plus compiled typed-walk sketches predicts the GNN's ranking effect within U_q on that dataset (an upper bound, not a full-pool claim)
- **l4_paths_flat**: the paths_over_attention interval reaches 0 or below: on that dataset the compiled typed-walk sketches add nothing measurable to one-hop attention. With a primary of L4_LOW, the fixed rungs through level 4 do not reproduce the GNN's ranking effect even with oracle access to its outputs.
- **objective_adds**: the objective_paths interval above 0 -- the objective limits recovery, not only the features
- **paths_node_local_adds**: the paths_over_node_local interval above 0 -- the sketches add to node-local features without any attention

### Level 3, as filed (bands and interpretation entries only)

| dataset | level 3 reading | level 3 interpretation |
|---|---|---|
| metaqa | L3_LOW | neighbourhood_adds, objective_adds, edge_effect_weighted |
| 2wiki | L3_HIGH | l3_high, l3_weighting_adds, neighbourhood_adds, objective_adds, edge_effect_weighted |
| squad | NOT_READ | none |

## metaqa

2,400 V2_GATE queries (the declared subsample: 800 per hop of the 19,738), 165,893 U_q rows (mean |U_q| 69.1); 25,831,324 message edges (10763 per query), 8.7 seeds per query; 4,313 U_q rows that no seed walk of length 1 to 3 reaches; largest walk counts 4.37e+03 (from the seeds) and 1.13e+03 (from anywhere); 1,967,659 one-hop entries equal to level 3's; sketch sha256 `426079a4e70ae405`.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.061 | [0.055, 0.068] | yes |
| full_coverage@5 | 0.065 | [0.055, 0.074] | yes |
| hit@1 | 0.127 | [0.114, 0.140] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L4_LOW |
| L4-mlp | 0.576 | 0.702 | 0.577 | 0.517 | 0.545 | -0.519 | -0.427 | -0.809 | -0.585 [-0.711, -0.477] | L4_LOW |
| L3-att | 0.690 | 0.781 | 0.649 | 0.626 | 0.602 | 0.025 | 0.034 | -0.287 | -0.076 [-0.161, 0.002] | L4_LOW |
| L4-att | 0.685 | 0.776 | 0.646 | 0.618 | 0.595 | -0.075 | -0.077 | -0.395 | -0.182 [-0.278, -0.097] | L4_LOW |
| L4-list | 0.397 | 0.594 | 0.511 | 0.663 | 0.578 | 0.310 | 0.316 | 0.039 | 0.222 [0.153, 0.289] | L4_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L4_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L4_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L3-att | 0.842 | 0.886 | 0.776 | 0.918 | 0.572 | 0.809 | 0.759 | 0.692 | 0.753 [0.738, 0.769] | L4_HIGH |
| L4-att | 0.840 | 0.886 | 0.784 | 0.923 | 0.577 | 0.821 | 0.775 | 0.696 | 0.764 [0.749, 0.779] | L4_HIGH |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L4_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L4_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | L4-mlp rho_bar | L3-att rho_bar | L4-att rho_bar | L4-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.039 [-1.842, -0.582] | -0.210 [-0.713, 0.077] | 0.228 [-0.167, 0.521] | 0.242 [-0.124, 0.503] | 0.401 [0.089, 0.655] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -1.049 [-1.466, -0.759] | -1.226 [-1.692, -0.900] | -0.650 [-1.022, -0.402] | -0.853 [-1.286, -0.560] | 0.029 [-0.173, 0.193] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.489 [-0.602, -0.386] | -0.451 [-0.567, -0.352] | 0.045 [-0.038, 0.120] | -0.049 [-0.135, 0.032] | 0.254 [0.177, 0.329] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.619 [-0.798, -0.474] | -0.471 [-0.632, -0.339] | 0.056 [-0.070, 0.156] | -0.044 [-0.178, 0.062] | 0.167 [0.051, 0.272] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -0.577 [-0.797, -0.418] | -0.613 [-0.840, -0.456] | -0.193 [-0.366, -0.065] | -0.303 [-0.488, -0.155] | 0.249 [0.136, 0.345] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -0.898 [-1.256, -0.631] | -0.840 [-1.185, -0.580] | -0.189 [-0.462, 0.006] | -0.301 [-0.570, -0.083] | 0.311 [0.151, 0.459] | 1.013 [0.971, 1.057] |

## 2wiki

6,290 V2_GATE queries (all of them), 245,055 U_q rows (mean |U_q| 39.0); 3,091,190 message edges (491 per query), 8.2 seeds per query; 40,852 U_q rows that no seed walk of length 1 to 3 reaches; largest walk counts 2.5e+04 (from the seeds) and 3.99e+03 (from anywhere); 1,580,750 one-hop entries equal to level 3's; sketch sha256 `149310c5912448b9`.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.048 | [0.045, 0.051] | yes |
| full_coverage@5 | 0.104 | [0.097, 0.111] | yes |
| hit@1 | -0.007 | [-0.012, -0.002] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L4_LOW |
| L4-mlp | 0.466 | 0.644 | 0.487 | 0.520 | 0.768 | 0.308 | 0.199 | -1.235 (nr) | 0.254 [0.203, 0.303] | L4_MID |
| L3-att | 0.645 | 0.753 | 0.596 | 0.665 | 0.816 | 0.831 | 0.767 | -0.432 (nr) | 0.799 [0.759, 0.838] | L4_HIGH |
| L4-att | 0.642 | 0.751 | 0.593 | 0.663 | 0.816 | 0.846 | 0.792 | -0.212 (nr) | 0.819 [0.774, 0.859] | L4_HIGH |
| L4-list | 0.563 | 0.707 | 0.569 | 0.664 | 0.814 | 0.911 | 0.871 | 0.235 (nr) | 0.891 [0.851, 0.928] | L4_HIGH |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L4_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L4_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L3-att | 0.782 | 0.809 | 0.712 | 0.853 | 0.834 | 0.841 | 0.792 | 1.937 (nr) | 0.816 [0.802, 0.829] | L4_HIGH |
| L4-att | 0.783 | 0.810 | 0.716 | 0.857 | 0.833 | 0.846 | 0.801 | 2.062 (nr) | 0.823 [0.810, 0.836] | L4_HIGH |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L4_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L4_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | L4-mlp rho_bar | L3-att rho_bar | L4-att rho_bar | L4-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | 0.193 [0.128, 0.255] | 0.455 [0.400, 0.512] | 0.948 [0.898, 0.997] | 0.953 [0.901, 1.005] | 0.989 [0.940, 1.037] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.232 [-0.343, -0.141] | -0.165 [-0.264, -0.078] | 0.491 [0.420, 0.562] | 0.542 [0.474, 0.608] | 0.689 [0.626, 0.750] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.567 [-0.025, 1.209] | 0.352 [-0.599, 0.960] | 0.238 [-0.548, 0.809] | 0.290 [-0.530, 0.820] | 0.475 [-0.089, 1.052] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | 0.033 [-0.024, 0.087] | 0.242 [0.189, 0.294] | 0.805 [0.764, 0.846] | 0.824 [0.781, 0.864] | 0.901 [0.862, 0.942] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.476 [0.267, 0.718] | 0.589 [0.377, 0.830] | 0.766 [0.528, 0.989] | 0.737 [0.511, 0.967] | 0.682 [0.452, 0.909] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a | n/a | n/a |

## squad

5,841 V2_GATE queries (all of them; the control), 176,880 U_q rows (mean |U_q| 30.3); 2,603,622 message edges (446 per query), 7.6 seeds per query; 33,109 U_q rows that no seed walk of length 1 to 3 reaches; largest walk counts 4.95e+04 (from the seeds) and 4.37e+03 (from anywhere); 1,905,184 one-hop entries equal to level 3's; sketch sha256 `1db752b2bdf16938`.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.004 | [-0.007, -0.002] | no |
| full_coverage@5 | -0.004 | [-0.007, -0.002] | no |
| hit@1 | -0.009 | [-0.013, -0.005] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.443 | 0.600 | 0.421 | 0.348 | 0.793 | -0.342 (nr) | -0.342 (nr) | -0.252 (nr) | n/a | NOT_READ |
| L4-mlp | 0.451 | 0.609 | 0.433 | 0.367 | 0.795 | -0.461 (nr) | -0.461 (nr) | -0.201 (nr) | n/a | NOT_READ |
| L3-att | 0.517 | 0.666 | 0.505 | 0.431 | 0.811 | 0.000 (nr) | 0.000 (nr) | -0.031 (nr) | n/a | NOT_READ |
| L4-att | 0.516 | 0.666 | 0.508 | 0.435 | 0.812 | 0.145 (nr) | 0.145 (nr) | 0.057 (nr) | n/a | NOT_READ |
| L4-list | 0.425 | 0.594 | 0.438 | 0.413 | 0.801 | -0.224 (nr) | -0.224 (nr) | 0.119 (nr) | n/a | NOT_READ |
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
| L3-att | 0.569 | 0.725 | 0.612 | 0.566 | 0.957 | 1.444 (nr) | 1.444 (nr) | 1.524 (nr) | n/a | NOT_READ |
| L4-att | 0.567 | 0.723 | 0.616 | 0.570 | 0.957 | 1.222 (nr) | 1.222 (nr) | 2.000 (nr) | n/a | NOT_READ |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | L4-mlp rho_bar | L3-att rho_bar | L4-att rho_bar | L4-list rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a | n/a | n/a |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity within U_q; it never describes a deployable model, and no probe output or sketch enters any retriever, feature, teacher or selection.
- A model that used these sketches without GNN outputs would be a competitor, and would need its own declaration under the QLS-U contract. This file does not open one.
- The hash buckets are dataset-specific in practice, and each dataset is probed on its own; nothing here transfers a sketch across datasets.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).
- No later level or arm is opened by any reading here.

## Reproducibility, placement and compute

| dataset | compile (min, laptop) | probes (min) | read (min) | device | driver | determinism warnings | mirror.json sha256 (host) | level 3 mirror.json sha256 (host) | repeat | refits bit-identical to level 3 |
|---|---:|---:|---:|---|---|---:|---|---|---|---|
| metaqa | 1.8 | 37 | 0.8 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `d35c3ba9fa4c204d` | `049d5cd2c3719895` | bit-identical | 3 of 3 |
| 2wiki | 2.1 | 48 | 1.8 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `0dbfa99f3c5986bf` | `4f2db84a5a1b0d80` | n/a | 3 of 3 |
| squad | 0.6 | 39 | 1.6 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `e4c32af527572e67` | `de7c1917942862d2` | n/a | 3 of 3 |

Placement: `host_gpu_det` = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": false, "env": "mpr-cu128@62fc45e9e1ba"}, applied by level 3's host_placement in every host process, with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified this file's mirror.json and level 3's against every file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was filed and checked against the committed files at the file stage. The compile ran on the laptop CPU at 6 threads from the same commit. Deviations: A 25-query metaqa smoke compile (--stage compile --limit 25, --out-dir in the session scratchpad) ran on the laptop at 2026-09-29T23:09:06Z, 32 s before a81451a was committed, from script bytes identical to that commit's (LF sha256 a12c8a19feeb5538..., git HEAD then 03ce076). It took 18 s and found 0 mismatches against level 3. Its output was discarded, no probe was fitted on it, and nothing of it enters the record.
