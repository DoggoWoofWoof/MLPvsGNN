# UMLP-D0.1: what the golds the twin misses look like (2wiki)

Analysis only (configs/umlp_d01_2wiki_feature_readout.yaml). 2wiki dev **V2_GATE**; queries where the twin's rank-1 item is a gold, formed per twin seed. Inference-only forward pass of the frozen twin (u_mlp_v2_mix s0-2) and GNN (u_gnn_v2_ef s0-2) checkpoints; every stored per-query metric reproduced exactly (6,290 queries, 0 mismatches). No fit, no new column, no held row. The M3B GAT was not loaded; 'MP recovers' means u_gnn_v2_ef (top 5 on >= 2 of 3 seeds).

Statistic: within-query pair-AUC, P(x > y) + 0.5 P(x = y) over pairs from the same query, averaged per query then over queries; query bootstrap, 1000 resamples, 95 %. 0.5 = indistinguishable; above 0.5 = the first group has larger values.

## Readings

| reading | verdict |
|---|---|
| R1_visible_but_scored_down | **INCONCLUSIVE** |
| R2_invisible_in_the_compiled_basis | **INCONCLUSIVE** |
| R3_compiled_but_unread | **NOT_SUPPORTED** |

R1 evidence (h1 stratum, C2 = missed gold vs displacing non-gold): read structural columns with CI above 0.60 per seed: s0 0; s1 0; s2 0; twin residual below 0.50 per seed: {'u_mlp_v2_mix__H128__s0': True, 'u_mlp_v2_mix__H128__s1': True, 'u_mlp_v2_mix__H128__s2': True}.
R2 evidence: GNN score C2 above 0.60 per seed: {'u_mlp_v2_mix__H128__s0': True, 'u_mlp_v2_mix__H128__s1': True, 'u_mlp_v2_mix__H128__s2': True}; read columns above 0.60 on all three seeds: 0.
R3 columns: none.

## Group sizes

| twin seed | analysed queries | G_top | G_rec | G_miss | G_miss_mp | G_miss_both | N_hard | N_disp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s0 | 5,728 | 5,728 | 6,267 | 1,424 | 911 | 513 | 44,383 | 16,645 |
| s1 | 5,557 | 5,557 | 6,149 | 1,364 | 829 | 535 | 42,952 | 16,079 |
| s2 | 5,657 | 5,657 | 6,163 | 1,487 | 964 | 523 | 43,785 | 16,465 |

## C2: missed gold vs displacing non-gold (ranks 2-5) -- h1 stratum, seed 0, score terms and the 20 columns furthest from 0.5

Queries with both groups: 1,209. Seeds 1-2 in the record.

| column | tier | AUC [95% CI] s0 | s1 | s2 |
|---|---|---|---|---|
| twin_score | score | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| twin_base | score | 0.394 [0.377, 0.413] | 0.399 [0.380, 0.418] | 0.452 [0.434, 0.470] |
| twin_residual | score | 0.218 [0.202, 0.234] | 0.132 [0.119, 0.145] | 0.160 [0.146, 0.174] |
| gnn_score | score | 0.646 [0.625, 0.667] | 0.624 [0.603, 0.645] | 0.664 [0.644, 0.684] |
| dense_cos | read | 0.390 [0.371, 0.410] | 0.390 [0.369, 0.411] | 0.433 [0.414, 0.452] |
| rrf | read | 0.394 [0.377, 0.413] | 0.399 [0.380, 0.418] | 0.452 [0.434, 0.470] |
| dense_rr | read | 0.395 [0.377, 0.413] | 0.395 [0.378, 0.414] | 0.448 [0.430, 0.466] |
| max_q_nbr_ner | - | 0.592 [0.574, 0.610] | 0.423 [0.405, 0.444] | 0.592 [0.573, 0.610] |
| splade_rr | read | 0.408 [0.392, 0.426] | 0.416 [0.399, 0.434] | 0.458 [0.440, 0.474] |
| splade_score_norm | - | 0.408 [0.392, 0.426] | 0.416 [0.399, 0.434] | 0.458 [0.440, 0.474] |
| gcs_full | read | 0.409 [0.391, 0.430] | 0.424 [0.404, 0.446] | 0.474 [0.456, 0.494] |
| deg_global_NER | read | 0.588 [0.568, 0.607] | 0.452 [0.430, 0.473] | 0.575 [0.555, 0.596] |
| gcs_struct | read | 0.414 [0.396, 0.433] | 0.424 [0.404, 0.445] | 0.481 [0.461, 0.500] |
| component_size_NER | read | 0.584 [0.568, 0.601] | 0.414 [0.396, 0.432] | 0.565 [0.548, 0.583] |
| dist1_NER | read | 0.584 [0.570, 0.600] | 0.472 [0.457, 0.487] | 0.586 [0.571, 0.601] |
| seedmass_h1_STRUCT | unread | 0.580 [0.559, 0.599] | 0.570 [0.551, 0.591] | 0.529 [0.508, 0.548] |
| has_nbr_ner | read | 0.577 [0.562, 0.591] | 0.416 [0.402, 0.432] | 0.566 [0.551, 0.581] |
| cos_q_proto_ner | read | 0.572 [0.554, 0.590] | 0.405 [0.386, 0.425] | 0.571 [0.552, 0.589] |
| cohesion_ner | read | 0.572 [0.554, 0.592] | 0.403 [0.384, 0.424] | 0.564 [0.545, 0.583] |
| support_h2_FULL | read | 0.428 [0.409, 0.449] | 0.423 [0.404, 0.444] | 0.395 [0.376, 0.416] |
| cos_v_seedproto_h2 | read | 0.429 [0.411, 0.448] | 0.403 [0.384, 0.423] | 0.389 [0.368, 0.408] |
| cos_v_seedproto | read | 0.431 [0.411, 0.453] | 0.388 [0.368, 0.408] | 0.429 [0.408, 0.449] |
| walks2_STRUCT | read | 0.433 [0.414, 0.452] | 0.418 [0.399, 0.439] | 0.404 [0.382, 0.424] |
| support_h1_STRUCT | unread | 0.567 [0.546, 0.587] | 0.559 [0.536, 0.580] | 0.526 [0.505, 0.548] |

## C1: missed gold vs recovered extra gold -- h1 stratum, seed 0, score terms and the 20 columns furthest from 0.5

Queries with both groups: 611. Seeds 1-2 in the record.

| column | tier | AUC [95% CI] s0 | s1 | s2 |
|---|---|---|---|---|
| twin_score | score | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| twin_base | score | 0.385 [0.351, 0.415] | 0.382 [0.353, 0.412] | 0.423 [0.395, 0.452] |
| twin_residual | score | 0.239 [0.214, 0.267] | 0.191 [0.168, 0.215] | 0.207 [0.183, 0.233] |
| gnn_score | score | 0.267 [0.240, 0.297] | 0.208 [0.180, 0.238] | 0.258 [0.230, 0.285] |
| support_h2_FULL | read | 0.320 [0.289, 0.348] | 0.316 [0.288, 0.343] | 0.314 [0.288, 0.342] |
| support_h2_STRUCT | read | 0.331 [0.302, 0.359] | 0.334 [0.306, 0.362] | 0.338 [0.313, 0.367] |
| cos_v_seedproto_h2 | read | 0.338 [0.308, 0.366] | 0.327 [0.298, 0.357] | 0.324 [0.297, 0.352] |
| deg_pool_STRUCT | read | 0.352 [0.322, 0.383] | 0.330 [0.301, 0.358] | 0.323 [0.295, 0.353] |
| ring_n_h1_STRUCT | unread | 0.352 [0.322, 0.383] | 0.330 [0.301, 0.358] | 0.323 [0.295, 0.353] |
| deg_pool_FULL | read | 0.357 [0.327, 0.387] | 0.333 [0.305, 0.363] | 0.335 [0.307, 0.366] |
| ring_n_h1_FULL | unread | 0.357 [0.327, 0.387] | 0.333 [0.305, 0.363] | 0.335 [0.307, 0.366] |
| branch_div_FULL | - | 0.357 [0.327, 0.388] | 0.331 [0.301, 0.361] | 0.330 [0.301, 0.360] |
| branch_h2_FULL | unread | 0.357 [0.327, 0.388] | 0.331 [0.301, 0.361] | 0.330 [0.301, 0.360] |
| cos_v_reachproto | read | 0.358 [0.327, 0.389] | 0.341 [0.311, 0.372] | 0.344 [0.316, 0.376] |
| walks2_FULL | read | 0.359 [0.328, 0.392] | 0.332 [0.301, 0.362] | 0.327 [0.300, 0.358] |
| paths_h2_FULL | unread | 0.359 [0.328, 0.392] | 0.332 [0.301, 0.362] | 0.327 [0.300, 0.358] |
| branch_div_STRUCT | - | 0.368 [0.338, 0.398] | 0.347 [0.316, 0.378] | 0.347 [0.320, 0.377] |
| gcs_struct | read | 0.368 [0.334, 0.401] | 0.382 [0.353, 0.414] | 0.413 [0.382, 0.442] |
| branch_h2_STRUCT | unread | 0.368 [0.338, 0.398] | 0.347 [0.316, 0.378] | 0.347 [0.320, 0.377] |
| walks2_STRUCT | read | 0.370 [0.339, 0.400] | 0.348 [0.318, 0.378] | 0.348 [0.320, 0.379] |
| paths_h2_STRUCT | unread | 0.370 [0.339, 0.400] | 0.348 [0.318, 0.378] | 0.348 [0.320, 0.379] |
| is_seed | read | 0.370 [0.343, 0.395] | 0.366 [0.343, 0.391] | 0.390 [0.367, 0.414] |
| dist0_STRUCT | - | 0.370 [0.343, 0.395] | 0.366 [0.343, 0.391] | 0.390 [0.367, 0.414] |
| dist0_NER | - | 0.370 [0.343, 0.395] | 0.366 [0.343, 0.391] | 0.390 [0.367, 0.414] |

## C4: missed golds the GNN recovers vs missed by both -- h1 stratum, seed 0, score terms and the 20 columns furthest from 0.5

Queries with both groups: 43. Seeds 1-2 in the record.

| column | tier | AUC [95% CI] s0 | s1 | s2 |
|---|---|---|---|---|
| twin_score | score | 0.721 [0.581, 0.861] | 0.806 [0.684, 0.908] | 0.685 [0.543, 0.805] |
| twin_base | score | 0.477 [0.349, 0.593] | 0.388 [0.276, 0.505] | 0.391 [0.277, 0.505] |
| twin_residual | score | 0.651 [0.512, 0.791] | 0.806 [0.704, 0.898] | 0.674 [0.543, 0.804] |
| gnn_score | score | 0.977 [0.930, 1.000] | 1.000 [1.000, 1.000] | 0.957 [0.891, 1.000] |
| ring_qmean_h3_FULL | read | 0.372 [0.244, 0.512] | 0.469 [0.347, 0.592] | 0.467 [0.337, 0.598] |
| ring_qmax_h3_FULL | read | 0.401 [0.279, 0.541] | 0.531 [0.398, 0.663] | 0.429 [0.315, 0.560] |
| deg_global_NER | read | 0.407 [0.267, 0.546] | 0.372 [0.250, 0.485] | 0.375 [0.255, 0.500] |
| support_h3_STRUCT | read | 0.593 [0.454, 0.721] | 0.582 [0.449, 0.714] | 0.565 [0.424, 0.707] |
| cohesion_structural | read | 0.581 [0.430, 0.710] | 0.510 [0.367, 0.643] | 0.446 [0.304, 0.587] |
| component_size_STRUCT | read | 0.419 [0.326, 0.512] | 0.490 [0.408, 0.571] | 0.413 [0.326, 0.500] |
| ring_qmean_h2_STRUCT | read | 0.419 [0.279, 0.558] | 0.439 [0.316, 0.571] | 0.348 [0.217, 0.478] |
| deg_global_STRUCT | read | 0.570 [0.419, 0.709] | 0.592 [0.459, 0.725] | 0.543 [0.402, 0.674] |
| cohesion_knn | read | 0.570 [0.448, 0.680] | 0.525 [0.423, 0.622] | 0.505 [0.413, 0.598] |
| ring_qmean_h1_STRUCT | read | 0.430 [0.279, 0.581] | 0.592 [0.449, 0.735] | 0.457 [0.304, 0.598] |
| ring_qmean_h1_FULL | read | 0.430 [0.279, 0.581] | 0.571 [0.429, 0.714] | 0.457 [0.315, 0.598] |
| splade_rr | read | 0.436 [0.314, 0.558] | 0.398 [0.286, 0.515] | 0.364 [0.250, 0.478] |
| splade_score_norm | - | 0.436 [0.314, 0.558] | 0.398 [0.286, 0.515] | 0.364 [0.250, 0.478] |
| dist1_KNN | read | 0.558 [0.477, 0.628] | 0.536 [0.480, 0.592] | 0.522 [0.446, 0.598] |
| deg_pool_KNN | read | 0.558 [0.442, 0.663] | 0.515 [0.413, 0.612] | 0.495 [0.402, 0.592] |
| support_h3_FULL | read | 0.558 [0.419, 0.698] | 0.571 [0.429, 0.704] | 0.641 [0.511, 0.772] |
| gcs_full | read | 0.442 [0.291, 0.570] | 0.408 [0.285, 0.541] | 0.391 [0.261, 0.522] |
| seeds_at_h3_FULL | read | 0.442 [0.337, 0.547] | 0.531 [0.418, 0.633] | 0.451 [0.353, 0.544] |
| agreement | read | 0.448 [0.337, 0.552] | 0.449 [0.352, 0.551] | 0.397 [0.299, 0.500] |
| seeds_2hop_frac_NER | read | 0.552 [0.454, 0.651] | 0.515 [0.429, 0.607] | 0.505 [0.418, 0.592] |

## C2 by stratum, seed 0 (score terms)

| stratum | queries | twin_base | twin_residual | gnn_score |
|---|---:|---|---|---|
| all | 1,293 | 0.398 [0.381, 0.414] | 0.205 [0.189, 0.220] | 0.627 [0.605, 0.647] |
| gold_count=2 | 664 | 0.334 [0.312, 0.357] | 0.214 [0.191, 0.236] | 0.621 [0.592, 0.646] |
| gold_count>=3 | 629 | 0.465 [0.438, 0.488] | 0.196 [0.174, 0.219] | 0.633 [0.603, 0.664] |
| h1 | 1,209 | 0.394 [0.377, 0.413] | 0.218 [0.202, 0.234] | 0.646 [0.625, 0.667] |
| h2 | 56 | 0.538 [0.469, 0.605] | 0.009 [0.000, 0.022] | 0.364 [0.283, 0.440] |
| h3 | 17 | 0.255 [0.137, 0.392] | 0.073 [0.000, 0.176] | 0.333 [0.186, 0.495] |
| none | 11 | 0.341 [0.136, 0.545] | 0.023 [0.000, 0.068] | 0.318 [0.136, 0.523] |

## Limits

- Univariate: a column that does not separate alone may separate jointly; that needs a fit, which this file bars.
- 'Evidence collapsed by aggregation' (Case C) is not decidable here: it needs a new compiled summary.
- The GNN reference is u_gnn_v2_ef, not the M3B GAT (not loaded).
