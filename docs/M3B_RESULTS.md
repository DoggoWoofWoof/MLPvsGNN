# M3B — the controlled comparison on the frozen contract

**Registered question.** After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?

Declaration: `configs/m3b_controlled_comparison.yaml` (status `RUN`). Arms: `qls_u_sota_v1` (QLS-U), `gat_universal_v1` (universal GAT), `gat_no_mp_v1` (GAT-NO-MP, the causal control). Eval populations are the headroom populations, whole; test splits were never read; the substrate is the read-only served package pinned by its freeze record. Nothing here is a scientific result on test data.

**At a glance — δ_MP on recall@5, universal GAT − GAT-NO-MP, seed 0, paired 95 % interval, with the section-6 band verdict:** metaqa +0.147 [+0.144, +0.150] (NOT_READ); squad -0.011 [-0.015, -0.008] (CONTROL); musique +0.018 [+0.008, +0.027] (READ); hotpotqa +0.030 [+0.026, +0.035] (READ); 2wiki +0.052 [+0.048, +0.055] (READ); webqsp +0.028 [+0.014, +0.044] (NOT_READ). The reading is section 9; nothing above is a result on test data.

## 1. The frozen candidate contract (`candidate_contract_frozen_2026_09_13`)

One pool per dataset, fixed before any weight existed. Ceilings are pool ceilings (K-aware `recall_ceiling@k`), measured on the eval population; `as compiled` is the same ceiling on the pools the models actually saw (the measured pool plus any retrieval seed it did not already contain). The reading column says whether the knee rule was applied over every cell or within the family the user's plan named (amendment 1, `ruled_reading`); the rule-as-filed pick is beside it either way.

| dataset | pool | reading | R-ceil@1 | R-ceil@5 | R-ceil@20 | fraction of attainable@5 | candidates (mean) | as compiled R-ceil@5 / cand. | rule-as-filed pick |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | ruled_reading | 0.5257 | 0.8094 | 0.9262 | 0.978 | 2017 | 0.8094 / 2017 | `equal_rrf_budget_200+FULL:h2_c25` (0.902 at 1290) |
| squad | `equal_rrf_budget_50` | rule_as_filed | 0.9798 | 0.9798 | 0.9798 | 0.980 | 50 | 0.9798 / 50 | `equal_rrf_budget_50` (0.980 at 50) |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | rule_as_filed | 0.4034 | 0.9207 | 0.9207 | 0.921 | 2092 | 0.9207 / 2092 | `equal_rrf_budget_200+FULL:h2_c25` (0.921 at 2092) |
| hotpotqa | `equal_rrf_budget_50+STRUCT:h1_c25` | rule_as_filed | 0.4958 | 0.9800 | 0.9800 | 0.980 | 94 | 0.9800 / 94 | `equal_rrf_budget_50+STRUCT:h1_c25` (0.980 at 94) |
| 2wiki | `equal_rrf_budget_50+STRUCT:h1_c25` | rule_as_filed | 0.4452 | 0.9636 | 0.9636 | 0.964 | 106 | 0.9637 / 106 | `equal_rrf_budget_50+STRUCT:h1_c25` (0.964 at 106) |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | ruled_reading | 0.5736 | 0.7649 | 0.8067 | 0.863 | 2120 | 0.7649 / 2120 | `dense_top200+FULL:h3_c25_v2000` (0.863 at 2120) |

WebQSP's corpus ceiling column is 0.5255 (gold resolution on the served corpus): a column, never averaged.

## 2. The information contract F(q, v) and the fixed base score

111 compiled columns (directions A-D plus retrieval and the fixed GCS propagation) were screened on the fit carves (25,663,066 candidate rows; duplicates on a 198,942-row stride sample, |Spearman| >= 0.98); 78 survive as `QLS_U_CORE_CONTRACT` (sha256 `ce584194a751cf3c…`). Nothing was dropped for weak signal.

| dropped column | reason |
|---|---|
| `dense_in` | later_member_of_duplicate_pair |
| `splade_in` | later_member_of_duplicate_pair |
| `splade_score_norm` | later_member_of_duplicate_pair |
| `seed_rank` | later_member_of_duplicate_pair |
| `dist0_STRUCT` | later_member_of_duplicate_pair |
| `reach_STRUCT` | later_member_of_duplicate_pair |
| `seeds_1hop_STRUCT` | later_member_of_duplicate_pair |
| `branch_div_STRUCT` | later_member_of_duplicate_pair |
| `dist0_NER` | later_member_of_duplicate_pair |
| `reach_NER` | later_member_of_duplicate_pair |
| `walks2_NER` | later_member_of_duplicate_pair |
| `branch_div_NER` | later_member_of_duplicate_pair |
| `dist0_KNN` | later_member_of_duplicate_pair |
| `reach_KNN` | later_member_of_duplicate_pair |
| `avail_KNN` | zero_variance_everywhere |
| `branch_div_KNN` | later_member_of_duplicate_pair |
| `dist0_FULL` | later_member_of_duplicate_pair |
| `reach_FULL` | later_member_of_duplicate_pair |
| `avail_FULL` | zero_variance_everywhere |
| `branch_div_FULL` | later_member_of_duplicate_pair |
| `wmax_seed_ner` | later_member_of_duplicate_pair |
| `wsum_seed_ner` | later_member_of_duplicate_pair |
| `wmax_seed_knn` | later_member_of_duplicate_pair |
| `wsum_seed_knn` | later_member_of_duplicate_pair |
| `max_q_nbr_ner` | later_member_of_duplicate_pair |
| `max_q_nbr_knn` | later_member_of_duplicate_pair |
| `cos_v_seedproto_h1` | later_member_of_duplicate_pair |
| `has_seed_h1` | later_member_of_duplicate_pair |
| `typed_available` | zero_variance_everywhere |
| `relmean_seed` | later_member_of_duplicate_pair |
| `has_typed_seed_edge` | later_member_of_duplicate_pair |
| `relchain2_mean` | later_member_of_duplicate_pair |
| `has_relchain2` | later_member_of_duplicate_pair |

Duplicate pairs (earlier member kept): `dense_rr`~`dense_in` (0.997), `splade_rr`~`splade_in` (0.998), `splade_rr`~`splade_score_norm` (0.999), `splade_in`~`splade_score_norm` (0.998), `is_seed`~`seed_rank` (1.000), `is_seed`~`dist0_STRUCT` (1.000), `is_seed`~`dist0_NER` (1.000), `is_seed`~`dist0_KNN` (1.000), `is_seed`~`dist0_FULL` (1.000), `seed_rank`~`dist0_STRUCT` (1.000), `seed_rank`~`dist0_NER` (1.000), `seed_rank`~`dist0_KNN` (1.000), `seed_rank`~`dist0_FULL` (1.000), `dist0_STRUCT`~`dist0_NER` (1.000), `dist0_STRUCT`~`dist0_KNN` (1.000), `dist0_STRUCT`~`dist0_FULL` (1.000), `dist1_STRUCT`~`seeds_1hop_STRUCT` (0.985), `distunreached_STRUCT`~`reach_STRUCT` (1.000), `walks2_STRUCT`~`branch_div_STRUCT` (0.991), `dist0_NER`~`dist0_KNN` (1.000), `dist0_NER`~`dist0_FULL` (1.000), `distunreached_NER`~`reach_NER` (1.000), `seeds_1hop_NER`~`wmax_seed_ner` (1.000), `seeds_1hop_NER`~`wsum_seed_ner` (1.000), `seeds_2hop_frac_NER`~`walks2_NER` (0.986), `seeds_2hop_frac_NER`~`branch_div_NER` (0.985), `walks2_NER`~`branch_div_NER` (1.000), `dist0_KNN`~`dist0_FULL` (1.000), `distunreached_KNN`~`reach_KNN` (1.000), `seeds_1hop_KNN`~`wmax_seed_knn` (1.000), `seeds_1hop_KNN`~`wsum_seed_knn` (1.000), `walks2_KNN`~`branch_div_KNN` (1.000), `distunreached_FULL`~`reach_FULL` (1.000), `seeds_1hop_FULL`~`cos_v_seedproto_h1` (0.997), `seeds_1hop_FULL`~`has_seed_h1` (0.999), `walks2_FULL`~`branch_div_FULL` (0.981), `wmax_seed_ner`~`wsum_seed_ner` (1.000), `wmax_seed_knn`~`wsum_seed_knn` (1.000), `cos_q_proto_ner`~`max_q_nbr_ner` (0.995), `cos_q_proto_knn`~`max_q_nbr_knn` (0.999), `cos_v_seedproto_h1`~`has_seed_h1` (0.998), `relmax_seed`~`relmean_seed` (1.000), `relmax_seed`~`has_typed_seed_edge` (1.000), `relmean_seed`~`has_typed_seed_edge` (1.000), `relchain2_max`~`relchain2_mean` (1.000), `relchain2_max`~`has_relchain2` (0.995), `relchain2_mean`~`has_relchain2` (0.995)

**Fixed base score:** `rrf` — the candidate with the highest macro select recall@5 ranked alone (dense_cos 0.4576, rrf 0.4725, gcs_full 0.4723); it enters every arm as `base_z` and is what every arm returns at step 0.

| dataset | train N | select | fit | univariate R@5 `dense_cos` | univariate R@5 `rrf` | univariate R@5 `gcs_full` |
|---|---|---|---|---|---|---|
| metaqa | 329,282 | 1,497 | 5,960 | 0.0115 | 0.0050 | 0.0050 |
| squad | 130,319 | 1,498 | 5,856 | 0.9111 | 0.9247 | 0.9247 |
| musique | 19,938 | 1,534 | 4,601 | 0.5197 | 0.5379 | 0.5377 |
| hotpotqa | 90,447 | 1,508 | 5,930 | 0.6708 | 0.6935 | 0.6935 |
| 2wiki | 167,454 | 1,496 | 5,928 | 0.5048 | 0.5899 | 0.5894 |
| webqsp | 1,549 | 310 | 1,239 | 0.1275 | 0.0837 | 0.0837 |

Strongest single columns by univariate select recall@5 (reported only, never used to drop): metaqa: `max_q_nbr_structural` 0.221, `relmax_seed` 0.158, `relmean_seed` 0.151, `splade_in` 0.143, `seeds_1hop_STRUCT` 0.119; squad: `gcs_full` 0.925, `gcs_struct` 0.925, `rrf` 0.925, `dense_cos` 0.911, `dense_rr` 0.910; musique: `rrf` 0.538, `gcs_struct` 0.538, `gcs_full` 0.538, `dense_rr` 0.520, `dense_cos` 0.520; hotpotqa: `gcs_full` 0.694, `rrf` 0.694, `gcs_struct` 0.693, `dense_rr` 0.671, `dense_cos` 0.671; 2wiki: `rrf` 0.590, `gcs_struct` 0.590, `gcs_full` 0.589, `splade_score_norm` 0.584, `splade_rr` 0.584; webqsp: `relmean_seed` 0.309, `relmax_seed` 0.297, `seed_edges_in` 0.203, `seeds_1hop_STRUCT` 0.201, `seed_edges_out` 0.192.

## 3. Selection behind the firewall

The GAT is chosen among GAT candidates only (L x H, seed 0, macro select recall@5, ties to the smaller); QLS-U among its own widths. No eval number existed when `outputs/m3b/selection.json` was written.

| arm | configuration | select macro R@5 | best epoch | seconds |
|---|---|---|---|---|
| universal GAT | L=2, H=64 | 0.8041 | 5 | 35214 |
| universal GAT | L=2, H=128 **(selected)** | 0.8109 | 5 | 34768 |
| universal GAT | L=3, H=64 | 0.8066 | 5 | 30420 |
| universal GAT | L=3, H=128 | 0.8099 | 3 | 36534 |
| QLS-U | H=64 | 0.7690 | 4 | 11917 |
| QLS-U | H=128 **(selected)** | 0.7729 | 4 | 19702 |
| GAT-NO-MP | L=2, H=128 (the GAT's) | — | — | — |

## 4. The controlled comparison: fixed compilation → QLS-U → GAT-NO-MP → universal GAT

Same F(q, v), same projections, same readout, same loss, optimiser and schedule; one universal set of weights per arm over the six fit carves. Seed 0 carries the paired statistic; the seed columns give mean ± sd over seeds 0-2 where they exist. `δ_MP` = universal GAT − GAT-NO-MP (the increment attributable to message passing itself, capacity and training held fixed); `GAT − QLS-U` is the question's other difference. Intervals: paired bootstrap over eval queries, 1000 resamples, `default_rng(0)`, 95 % percentile.

### metaqa — eval `dev`, 39,138 queries, pool `equal_rrf_budget_50+STRUCT:h3_c25_v2000` (R-ceil@5 0.8094, 2017 candidates)

| metric | fixed `rrf` | best fixed (dense_cos) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.0003 | 0.0007 | 0.3457 | 0.3453 ± 0.0003 (n=3) | 0.3416 | 0.3450 ± 0.0028 (n=3) | 0.4516 | 0.4541 ± 0.0017 (n=3) | +0.110 [+0.107, +0.114] | +0.106 [+0.103, +0.109] | 0.5257 |
| recall@5 | 0.0050 | 0.0085 | 0.6124 | 0.6135 ± 0.0010 (n=3) | 0.6165 | 0.6179 ± 0.0012 (n=3) | 0.7634 | 0.7645 ± 0.0011 (n=3) | +0.147 [+0.144, +0.150] | +0.151 [+0.148, +0.154] | 0.8094 |
| recall@20 | 0.0156 | 0.0200 | 0.7799 | 0.7801 ± 0.0002 (n=3) | 0.7832 | 0.7825 ± 0.0012 (n=3) | 0.9014 | 0.9024 ± 0.0013 (n=3) | +0.118 [+0.116, +0.121] | +0.122 [+0.119, +0.124] | 0.9262 |
| mrr | 0.0159 | 0.0214 | 0.7129 | 0.7117 ± 0.0016 (n=3) | 0.7109 | 0.7133 ± 0.0017 (n=3) | 0.8869 | 0.8937 ± 0.0059 (n=3) | +0.176 [+0.173, +0.180] | +0.174 [+0.170, +0.178] | — |
| hit@1 | 0.0007 | 0.0019 | 0.6124 | 0.6108 ± 0.0018 (n=3) | 0.6111 | 0.6134 ± 0.0021 (n=3) | 0.8310 | 0.8411 ± 0.0091 (n=3) | +0.220 [+0.215, +0.225] | +0.219 [+0.213, +0.224] | — |
| full_coverage@5 | 0.0029 | 0.0047 | 0.4928 | 0.4946 ± 0.0013 (n=3) | 0.4972 | 0.4993 ± 0.0018 (n=3) | 0.6413 | 0.6414 ± 0.0002 (n=3) | +0.144 [+0.141, +0.148] | +0.149 [+0.145, +0.152] | 0.6969 |
| full_coverage@20 | 0.0090 | 0.0108 | 0.6806 | 0.6811 ± 0.0006 (n=3) | 0.6881 | 0.6847 ± 0.0027 (n=3) | 0.8203 | 0.8192 ± 0.0010 (n=3) | +0.132 [+0.129, +0.135] | +0.140 [+0.136, +0.143] | 0.8540 |

### squad — eval `dev`, 11,873 queries, pool `equal_rrf_budget_50` (R-ceil@5 0.9798, 50 candidates)

| metric | fixed `rrf` | best fixed (rrf) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.7349 | 0.7349 | 0.7293 | 0.7216 ± 0.0107 (n=3) | 0.7360 | 0.7369 ± 0.0023 (n=3) | 0.7200 | 0.7304 ± 0.0079 (n=3) | -0.016 [-0.021, -0.011] | -0.009 [-0.014, -0.004] | 0.9798 |
| recall@5 | 0.9053 | 0.9053 | 0.8894 | 0.8867 ± 0.0123 (n=3) | 0.8959 | 0.8973 ± 0.0043 (n=3) | 0.8846 | 0.8922 ± 0.0061 (n=3) | -0.011 [-0.015, -0.008] | -0.005 [-0.009, -0.001] | 0.9798 |
| recall@20 | 0.9640 | 0.9640 | 0.9533 | 0.9530 ± 0.0051 (n=3) | 0.9594 | 0.9609 ± 0.0021 (n=3) | 0.9541 | 0.9569 ± 0.0023 (n=3) | -0.005 [-0.008, -0.003] | +0.001 [-0.002, +0.003] | 0.9798 |
| mrr | 0.8114 | 0.8114 | 0.8019 | 0.7965 ± 0.0106 (n=3) | 0.8084 | 0.8093 ± 0.0032 (n=3) | 0.7944 | 0.8035 ± 0.0070 (n=3) | -0.014 [-0.017, -0.011] | -0.007 [-0.011, -0.004] | — |
| hit@1 | 0.7349 | 0.7349 | 0.7293 | 0.7216 ± 0.0107 (n=3) | 0.7360 | 0.7369 ± 0.0023 (n=3) | 0.7200 | 0.7304 ± 0.0079 (n=3) | -0.016 [-0.021, -0.011] | -0.009 [-0.014, -0.004] | — |
| full_coverage@5 | 0.9053 | 0.9053 | 0.8894 | 0.8867 ± 0.0123 (n=3) | 0.8959 | 0.8973 ± 0.0043 (n=3) | 0.8846 | 0.8922 ± 0.0061 (n=3) | -0.011 [-0.015, -0.008] | -0.005 [-0.009, -0.001] | 0.9798 |
| full_coverage@20 | 0.9640 | 0.9640 | 0.9533 | 0.9530 ± 0.0051 (n=3) | 0.9594 | 0.9609 ± 0.0021 (n=3) | 0.9541 | 0.9569 ± 0.0023 (n=3) | -0.005 [-0.008, -0.003] | +0.001 [-0.002, +0.003] | 0.9798 |

### musique — eval `dev`, 2,417 queries, pool `equal_rrf_budget_200+FULL:h2_c25` (R-ceil@5 0.9207, 2092 candidates)

| metric | fixed `rrf` | best fixed (rrf) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.2606 | 0.2606 | 0.2431 | 0.2397 ± 0.0090 (n=3) | 0.2616 | 0.2480 ± 0.0115 (n=3) | 0.2673 | 0.2671 ± 0.0005 (n=3) | +0.006 [-0.001, +0.012] | +0.024 [+0.018, +0.031] | 0.4034 |
| recall@5 | 0.4734 | 0.4734 | 0.4912 | 0.4884 ± 0.0043 (n=3) | 0.5030 | 0.5049 ± 0.0048 (n=3) | 0.5207 | 0.5289 ± 0.0069 (n=3) | +0.018 [+0.008, +0.027] | +0.029 [+0.020, +0.038] | 0.9207 |
| recall@20 | 0.6189 | 0.6189 | 0.6375 | 0.6485 ± 0.0132 (n=3) | 0.6556 | 0.6601 ± 0.0043 (n=3) | 0.6815 | 0.6902 ± 0.0080 (n=3) | +0.026 [+0.018, +0.034] | +0.044 [+0.036, +0.052] | 0.9207 |
| mrr | 0.7388 | 0.7388 | 0.7074 | 0.7026 ± 0.0148 (n=3) | 0.7456 | 0.7219 ± 0.0199 (n=3) | 0.7452 | 0.7489 ± 0.0027 (n=3) | -0.000 [-0.011, +0.010] | +0.038 [+0.027, +0.049] | — |
| hit@1 | 0.6347 | 0.6347 | 0.5912 | 0.5827 ± 0.0218 (n=3) | 0.6409 | 0.6054 ± 0.0296 (n=3) | 0.6504 | 0.6529 ± 0.0018 (n=3) | +0.010 [-0.007, +0.026] | +0.059 [+0.043, +0.077] | — |
| full_coverage@5 | 0.1353 | 0.1353 | 0.1800 | 0.1750 ± 0.0035 (n=3) | 0.1746 | 0.1860 ± 0.0090 (n=3) | 0.2238 | 0.2322 ± 0.0079 (n=3) | +0.049 [+0.036, +0.063] | +0.044 [+0.031, +0.056] | 0.8026 |
| full_coverage@20 | 0.2772 | 0.2772 | 0.3422 | 0.3532 ± 0.0153 (n=3) | 0.3517 | 0.3635 ± 0.0102 (n=3) | 0.3951 | 0.4099 ± 0.0135 (n=3) | +0.043 [+0.030, +0.058] | +0.053 [+0.038, +0.067] | 0.8026 |

### hotpotqa — eval `validation`, 7,405 queries, pool `equal_rrf_budget_50+STRUCT:h1_c25` (R-ceil@5 0.9800, 94 candidates)

| metric | fixed `rrf` | best fixed (gcs_full) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.4145 | 0.4145 | 0.4123 | 0.4125 ± 0.0016 (n=3) | 0.4244 | 0.4216 ± 0.0035 (n=3) | 0.4192 | 0.4229 ± 0.0034 (n=3) | -0.005 [-0.009, -0.001] | +0.007 [+0.003, +0.011] | 0.4958 |
| recall@5 | 0.6845 | 0.6849 | 0.8703 | 0.8721 ± 0.0027 (n=3) | 0.8658 | 0.8744 ± 0.0062 (n=3) | 0.8959 | 0.8975 ± 0.0044 (n=3) | +0.030 [+0.026, +0.035] | +0.026 [+0.021, +0.030] | 0.9800 |
| recall@20 | 0.7862 | 0.7886 | 0.9518 | 0.9530 ± 0.0018 (n=3) | 0.9458 | 0.9522 ± 0.0047 (n=3) | 0.9601 | 0.9609 ± 0.0011 (n=3) | +0.014 [+0.011, +0.017] | +0.008 [+0.006, +0.011] | 0.9800 |
| mrr | 0.8815 | 0.8815 | 0.8863 | 0.8870 ± 0.0018 (n=3) | 0.8996 | 0.8973 ± 0.0035 (n=3) | 0.8913 | 0.8969 ± 0.0044 (n=3) | -0.008 [-0.013, -0.004] | +0.005 [+0.000, +0.010] | — |
| hit@1 | 0.8290 | 0.8290 | 0.8246 | 0.8250 ± 0.0032 (n=3) | 0.8488 | 0.8432 ± 0.0070 (n=3) | 0.8384 | 0.8458 ± 0.0069 (n=3) | -0.010 [-0.018, -0.003] | +0.014 [+0.006, +0.022] | — |
| full_coverage@5 | 0.4240 | 0.4247 | 0.7787 | 0.7819 ± 0.0052 (n=3) | 0.7693 | 0.7856 ± 0.0118 (n=3) | 0.8377 | 0.8367 ± 0.0074 (n=3) | +0.068 [+0.060, +0.077] | +0.059 [+0.050, +0.067] | 0.9684 |
| full_coverage@20 | 0.5938 | 0.5986 | 0.9192 | 0.9229 ± 0.0040 (n=3) | 0.9079 | 0.9211 ± 0.0095 (n=3) | 0.9387 | 0.9391 ± 0.0020 (n=3) | +0.031 [+0.026, +0.036] | +0.019 [+0.015, +0.024] | 0.9684 |

### 2wiki — eval `dev`, 12,576 queries, pool `equal_rrf_budget_50+STRUCT:h1_c25` (R-ceil@5 0.9637, 106 candidates)

| metric | fixed `rrf` | best fixed (gcs_full) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.3880 | 0.3880 | 0.3863 | 0.3840 ± 0.0044 (n=3) | 0.4001 | 0.3958 ± 0.0044 (n=3) | 0.3899 | 0.3920 ± 0.0037 (n=3) | -0.010 [-0.013, -0.008] | +0.004 [+0.001, +0.007] | 0.4452 |
| recall@5 | 0.6079 | 0.6080 | 0.8377 | 0.8366 ± 0.0019 (n=3) | 0.8335 | 0.8389 ± 0.0041 (n=3) | 0.8850 | 0.8836 ± 0.0040 (n=3) | +0.052 [+0.048, +0.055] | +0.047 [+0.044, +0.051] | 0.9637 |
| recall@20 | 0.6946 | 0.6956 | 0.9405 | 0.9400 ± 0.0019 (n=3) | 0.9373 | 0.9413 ± 0.0029 (n=3) | 0.9517 | 0.9513 ± 0.0012 (n=3) | +0.014 [+0.013, +0.016] | +0.011 [+0.009, +0.013] | 0.9637 |
| mrr | 0.9243 | 0.9243 | 0.9235 | 0.9216 ± 0.0058 (n=3) | 0.9420 | 0.9362 ± 0.0062 (n=3) | 0.9266 | 0.9296 ± 0.0047 (n=3) | -0.015 [-0.018, -0.012] | +0.003 [-0.000, +0.007] | — |
| hit@1 | 0.8754 | 0.8754 | 0.8719 | 0.8679 ± 0.0100 (n=3) | 0.9037 | 0.8942 ± 0.0101 (n=3) | 0.8832 | 0.8867 ± 0.0086 (n=3) | -0.021 [-0.026, -0.015] | +0.011 [+0.005, +0.018] | — |
| full_coverage@5 | 0.2537 | 0.2537 | 0.6444 | 0.6406 ± 0.0028 (n=3) | 0.6306 | 0.6430 ± 0.0097 (n=3) | 0.7564 | 0.7495 ± 0.0072 (n=3) | +0.126 [+0.118, +0.133] | +0.112 [+0.105, +0.119] | 0.9059 |
| full_coverage@20 | 0.3826 | 0.3842 | 0.8589 | 0.8571 ± 0.0030 (n=3) | 0.8508 | 0.8601 ± 0.0065 (n=3) | 0.8845 | 0.8827 ± 0.0023 (n=3) | +0.034 [+0.030, +0.038] | +0.026 [+0.022, +0.029] | 0.9059 |

### webqsp — eval `train_holdout`, 1,503 queries, pool `dense_top200+FULL:h3_c25_v2000` (R-ceil@5 0.7649, 2120 candidates)

| metric | fixed `rrf` | best fixed (dense_cos) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.0242 | 0.0456 | 0.3064 | 0.3086 ± 0.0052 (n=3) | 0.3085 | 0.3124 ± 0.0028 (n=3) | 0.3388 | 0.3397 ± 0.0031 (n=3) | +0.030 [+0.014, +0.048] | +0.032 [+0.016, +0.049] | 0.5736 |
| recall@5 | 0.0535 | 0.1095 | 0.5768 | 0.5751 ± 0.0021 (n=3) | 0.5738 | 0.5765 ± 0.0051 (n=3) | 0.6019 | 0.6073 ± 0.0049 (n=3) | +0.028 [+0.014, +0.044] | +0.025 [+0.011, +0.041] | 0.7649 |
| recall@20 | 0.1012 | 0.1775 | 0.7231 | 0.7167 ± 0.0054 (n=3) | 0.6953 | 0.7086 ± 0.0099 (n=3) | 0.7155 | 0.7198 ± 0.0039 (n=3) | +0.020 [+0.010, +0.032] | -0.008 [-0.018, +0.003] | 0.8067 |
| mrr | 0.0767 | 0.1220 | 0.6452 | 0.6454 ± 0.0003 (n=3) | 0.6420 | 0.6474 ± 0.0038 (n=3) | 0.6818 | 0.6826 ± 0.0056 (n=3) | +0.040 [+0.025, +0.056] | +0.037 [+0.022, +0.051] | — |
| hit@1 | 0.0452 | 0.0712 | 0.5403 | 0.5438 ± 0.0030 (n=3) | 0.5436 | 0.5491 ± 0.0039 (n=3) | 0.5948 | 0.5928 ± 0.0096 (n=3) | +0.051 [+0.029, +0.074] | +0.055 [+0.031, +0.077] | — |
| full_coverage@5 | 0.0379 | 0.0878 | 0.4591 | 0.4573 ± 0.0021 (n=3) | 0.4611 | 0.4624 ± 0.0060 (n=3) | 0.4837 | 0.4875 ± 0.0053 (n=3) | +0.023 [+0.007, +0.040] | +0.025 [+0.009, +0.042] | 0.6587 |
| full_coverage@20 | 0.0752 | 0.1397 | 0.5948 | 0.5902 ± 0.0048 (n=3) | 0.5722 | 0.5839 ± 0.0087 (n=3) | 0.5888 | 0.5921 ± 0.0030 (n=3) | +0.017 [+0.003, +0.031] | -0.006 [-0.019, +0.007] | 0.6873 |

Corpus ceiling column: 0.5255 (reference level; every webqsp recall above is bounded by the served gold resolution).

### 4b. Select carve → eval population, the fixed base as the population control

The select carve is a slice of the train split held out from the fit carve; it chose the epoch and the configuration (section 3) and is not a result. For each arm (seed 0, the selected weights): recall@5 on the select carve at the best epoch (fit record) → on the eval population (section 4), and the same pair for the fixed base `rrf`, which has no parameters and moves only with the population. `shift` = eval − select; `beyond base` = the arm's shift minus the base's shift, i.e. the part of the arm's select-carve advantage over the fixed base that does not carry to the eval population. The last column (`scripts/m3b_population_overlap.py`, golds resolved by the headroom's function) is the fraction of each population's queries that hold at least one gold node which is a gold node of some fit-carve query: where the select carve shares fit golds and the eval population does not, the select carve was not distribution-matched with the eval population and its learned lift is read as train-internal. Reporting only; no rule reads it.

| dataset | `rrf` select → eval (shift) | QLS-U select → eval (shift; beyond base) | GAT-NO-MP select → eval (shift; beyond base) | universal GAT select → eval (shift; beyond base) | queries holding a fit gold: select / eval |
|---|---|---|---|---|---|
| metaqa | 0.0050 → 0.0050 (-0.000) | 0.6259 → 0.6124 (-0.013; -0.013) | 0.6403 → 0.6165 (-0.024; -0.024) | 0.7789 → 0.7634 (-0.016; -0.015) | 0.828 / 0.864 |
| squad | 0.9247 → 0.9053 (-0.019) | 0.9059 → 0.8894 (-0.016; +0.003) | 0.9132 → 0.8959 (-0.017; +0.002) | 0.8972 → 0.8846 (-0.013; +0.007) | 0.290 / 0.000 |
| musique | 0.5379 → 0.4734 (-0.064) | 0.7479 → 0.4912 (-0.257; -0.192) | 0.7448 → 0.5030 (-0.242; -0.177) | 0.7611 → 0.5207 (-0.240; -0.176) | 0.945 / 0.000 (single-hop components shared with a fit question: 0.941 / 0.000) |
| hotpotqa | 0.6935 → 0.6845 (-0.009) | 0.8733 → 0.8703 (-0.003; +0.006) | 0.8740 → 0.8658 (-0.008; +0.001) | 0.8982 → 0.8959 (-0.002; +0.007) | 0.163 / 0.183 |
| 2wiki | 0.5899 → 0.6079 (+0.018) | 0.8730 → 0.8377 (-0.035; -0.053) | 0.8778 → 0.8335 (-0.044; -0.062) | 0.9054 → 0.8850 (-0.020; -0.038) | 0.428 / 0.308 |
| webqsp | 0.0837 → 0.0535 (-0.030) | 0.6115 → 0.5768 (-0.035; -0.005) | 0.5972 → 0.5738 (-0.023; +0.007) | 0.6246 → 0.6019 (-0.023; +0.007) | 0.528 / 0.517 |

## 5. The three substrates: message edges restricted to one family

The selected GAT retrained (seed 0) with its message edges restricted to STRUCT, NER or KNN; F(q, v) unchanged, so the input block still reads every family's fixed prototypes. Cells are recall@5 and the paired difference to GAT-NO-MP; on a dataset where the headroom showed a family adds no exposure (squad, every family) the cell is a control, not a regime result. Each substrate cell is a single fit (seed 0): a δ smaller than the full GAT's seed-to-seed spread on that dataset (last column, sd over seeds 0-2 from section 4) is not read.

| dataset | GAT-NO-MP | GAT[STRUCT] | δ | GAT[NER] | δ | GAT[KNN] | δ | GAT[FULL] | δ_MP | GAT seed sd |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.6165 | 0.7605 | +0.144 [+0.141, +0.147] | 0.6124 | -0.004 [-0.006, -0.002] | 0.6199 | +0.003 [+0.002, +0.005] | 0.7634 | +0.147 [+0.144, +0.150] | 0.0011 (n=3) |
| squad | 0.8959 | 0.9064 | +0.011 [+0.007, +0.014] | 0.9063 | +0.010 [+0.008, +0.014] | 0.8881 | -0.008 [-0.011, -0.004] | 0.8846 | -0.011 [-0.015, -0.008] | 0.0061 (n=3) |
| musique | 0.5030 | 0.5258 | +0.023 [+0.014, +0.031] | 0.5210 | +0.018 [+0.009, +0.026] | 0.5084 | +0.005 [-0.004, +0.014] | 0.5207 | +0.018 [+0.008, +0.027] | 0.0069 (n=3) |
| hotpotqa | 0.8658 | 0.8979 | +0.032 [+0.028, +0.036] | 0.8777 | +0.012 [+0.008, +0.016] | 0.8685 | +0.003 [-0.001, +0.007] | 0.8959 | +0.030 [+0.026, +0.035] | 0.0044 (n=3) |
| 2wiki | 0.8335 | 0.8884 | +0.055 [+0.051, +0.058] | 0.8543 | +0.021 [+0.018, +0.024] | 0.8444 | +0.011 [+0.008, +0.014] | 0.8850 | +0.052 [+0.048, +0.055] | 0.0040 (n=3) |
| webqsp | 0.5738 | 0.5957 | +0.022 [+0.007, +0.037] | 0.5612 | -0.013 [-0.026, +0.001] | 0.5619 | -0.012 [-0.024, +0.001] | 0.6019 | +0.028 [+0.014, +0.044] | 0.0049 (n=3) |

## 6. Calibration against the published systems (exposure attached)

Published numbers are read against their named exposure, never averaged with ours. Our number is given in the same form: recall@5 / recall@10 for the GraphER rows (their PR@K is gold-passage coverage at K on a 200-candidate query-induced corpus with per-dataset training), hit@1 for the KB rows (NuTrea / ReaRev: assigned topic entities, full KB neighbourhood up to 3 hops, per-dataset training). Ours: the frozen pool under the knee rule, seeds from retrieval, six-dataset shared weights, fit carve capped at 6000. The gap is read against these named differences, never averaged across them.

| dataset | published (exposure) | ours: universal GAT s0 | ours: QLS-U s0 | ours: GAT-NO-MP s0 | band check (universal GAT s0) | δ_MP |
|---|---|---|---|---|---|---|
| metaqa | NuTrea Hit@1 1-hop 97.4 / 2-hop 99.99 / 3-hop 98.89; ReaRev 3-hop 98.9 (KB exposure) | hit@1 1hop 0.888 / 2hop 0.882 / 3hop 0.737 (all 0.831) | hit@1 1hop 0.860 / 2hop 0.640 / 3hop 0.410 (all 0.612) | hit@1 1hop 0.842 / 2hop 0.653 / 3hop 0.406 (all 0.611) | 1hop 0.888 vs 0.974 (below by 0.086; measured exposure shortfall 0.002); 2hop 0.882 vs 1.000 (below by 0.118; measured exposure shortfall 0.002); 3hop 0.737 vs 0.989 (below by 0.252; measured exposure shortfall 0.002) | NOT_READ |
| squad | no graph-retrieval SOTA in the archaeology; the retrieval ceiling row stands in | R@5 0.885 | R@5 0.889 | R@5 0.896 | no published graph-retrieval number for this dataset; the retrieval ceiling row stands in | CONTROL |
| musique | GraphER PR@5 25.4 / 25.6 / 21.6; PR@10 36.9 / 37.4 / 32.4 (GraphER exposure) | R@5 0.521 / R@10 0.605 | R@5 0.491 / R@10 0.572 | R@5 0.503 / R@10 0.582 | recall@5 0.521 vs published 0.216-0.256 | READ |
| hotpotqa | GraphER PR@5 GCS 78.8 / GAT 78.0 / MLP 78.9; PR@10 88.7 / 88.9 / 87.2 (GraphER exposure) | R@5 0.896 / R@10 0.937 | R@5 0.870 / R@10 0.922 | R@5 0.866 / R@10 0.917 | recall@5 0.896 vs published 0.780-0.789 | READ |
| 2wiki | GraphER PR@5 43.8 / 44.1 / 42.5; PR@10 51.1 / 53.0 / 51.1 (GraphER exposure) | R@5 0.885 / R@10 0.933 | R@5 0.838 / R@10 0.906 | R@5 0.833 / R@10 0.902 | recall@5 0.885 vs published 0.425-0.441 | READ |
| webqsp | NuTrea Hit@1 77.43; ReaRev 76.4 (KB exposure; corpus ceiling 0.5255) | hit@1 0.595 | hit@1 0.540 | hit@1 0.544 | hit@1 0.595 vs published 0.764-0.774 (below by 0.169; measured exposure shortfall 0.027) | NOT_READ |

configs/m3a_compilation.yaml#universal_gnn_calibration_rule .when_delta_mp_is_not_read applies: if the universal GAT falls outside the SOTA band and no named exposure difference explains the gap, that cell reports the numbers and marks delta_MP NOT_READ.

**How the band check is applied.** inside when the universal GAT (seed 0) is at or above the lowest published number minus the measured exposure shortfall (published coverage minus our any_gold_at_pool); the differences that remain named but unmeasured here (training scale and universality, the fit cap, the node-text regime) do not lift a cell into the band. READ: δ_MP is read for that cell. NOT_READ: the numbers stand and δ_MP is reported as a measurement on an arm below the published band, not as the answer to the question for that cell; what would lift it is a per-dataset fit at the published training scale, which is outside this phase. CONTROL: no published number to calibrate against.

**How each system enters the graph, and what its candidate set exposes.** Published coverage is at-least-one-answer-in-the-extracted-subgraph (ReaRev Table 6 / GNN-RAG Table 7); our like-for-like figure is `any_gold_at_pool` of the frozen pool (not the corpus query-level coverage, which has no pool behind it, and not the pool ceiling@5, which is K-aware). The three arms share every row of ours: identical pool, identical features, identical base score, so a difference between them is a difference of the scorer.

| method | graph entry point | avg graph / pool | any-answer exposure | retrieval prior retained? | ranking metric |
|---|---|---|---|---|---|
| ReaRev (EMNLP Findings 2022) | provided topic entities + PageRank-Nibble top-m (He et al. 2021) | WebQSP 1,429.8 entities (m = 2,000); MetaQA-3 497.9 (m = 500) | WebQSP 94.9 %; MetaQA-3 99.0 % (Table 6) | no -- no dense / sparse retriever in the pipeline; the question enters through the entity | Hits@1 / F1 over answer entities |
| GNN-RAG (2024) | linked entities + PageRank-Nibble top-2,000 (same preprocessing family; ReaRev is its GNN) | WebQSP 1,429.8; MetaQA-3 497.9 (Table 7) | WebQSP 94.9 %; MetaQA-3 99.0 % (Table 7) | no -- the graph entry is the linked entity; its RA variant unions LLM-retrieved paths, a different stage | answer Hits@1 / F1 after the LLM; the GNN's own retrieval recall is reported separately |
| NuTrea (NeurIPS 2023) | assigned topic entities, 2-hop subgraph (archaeology row; its statistics were not re-read here) | NOT STATED | NOT STATED | no | Hits@1 / F1 |
| GraphER | the base retriever's top-200 passages; the graph is candidate-induced at query time | 200 candidates | not reported in that form; its PR@K is coverage at K | yes -- the retrieval score is an input to the GAT and to GCS; the closest comparator to this phase | PR@5 / PR@10 |
| **ours (metaqa): QLS-U = universal GAT = GAT-NO-MP** | Dense + SPLADE seeds (dense top-5 ∪ splade top-5); base pool `equal_rrf_budget_50`; expansion `STRUCT:h3_c25_v2000` | 2,017.0 candidates | any-gold 0.9876, all-gold 0.8938, pool ceiling@5 0.8094 | yes — retrieval columns in F(q, v) and the fixed base score in the readout | hit@1, R@5 |
| **ours (squad): QLS-U = universal GAT = GAT-NO-MP** | Dense + SPLADE seeds (dense top-5 ∪ splade top-5); base pool `equal_rrf_budget_50`; no graph expansion (retrieval-only pool) | 50.0 candidates | any-gold 0.9798, all-gold 0.9798, pool ceiling@5 0.9798 | yes — retrieval columns in F(q, v) and the fixed base score in the readout | R@5 / R@10 |
| **ours (musique): QLS-U = universal GAT = GAT-NO-MP** | Dense + SPLADE seeds (dense top-5 ∪ splade top-5); base pool `equal_rrf_budget_200`; expansion `FULL:h2_c25` | 2,092.4 candidates | any-gold 0.9950, all-gold 0.8026, pool ceiling@5 0.9207 | yes — retrieval columns in F(q, v) and the fixed base score in the readout | R@5 / R@10 |
| **ours (hotpotqa): QLS-U = universal GAT = GAT-NO-MP** | Dense + SPLADE seeds (dense top-5 ∪ splade top-5); base pool `equal_rrf_budget_50`; expansion `STRUCT:h1_c25` | 93.7 candidates | any-gold 0.9916, all-gold 0.9684, pool ceiling@5 0.9800 | yes — retrieval columns in F(q, v) and the fixed base score in the readout | R@5 / R@10 |
| **ours (2wiki): QLS-U = universal GAT = GAT-NO-MP** | Dense + SPLADE seeds (dense top-5 ∪ splade top-5); base pool `equal_rrf_budget_50`; expansion `STRUCT:h1_c25` | 105.6 candidates | any-gold 0.9997, all-gold 0.9058, pool ceiling@5 0.9636 | yes — retrieval columns in F(q, v) and the fixed base score in the readout | R@5 / R@10 |
| **ours (webqsp): QLS-U = universal GAT = GAT-NO-MP** | Dense + SPLADE seeds (dense top-5 ∪ splade top-5); base pool `dense_top200`; expansion `FULL:h3_c25_v2000` | 2,119.6 candidates | any-gold 0.9215, all-gold 0.6880, pool ceiling@5 0.7649 — below the 0.90 knee (fraction of attainable@5 0.8628; fallback rule) | yes — retrieval columns in F(q, v) and the fixed base score in the readout | hit@1, R@5 |

"Prior KGQA systems commonly construct question-specific subgraphs from linked topic entities, whereas our graph entry points are obtained from inference-safe dense and sparse retrieval. We therefore measure candidate headroom explicitly before model comparison. Despite the different entry mechanism, the resulting retrieval-seeded graphs attain high answer exposure, allowing us to study ranking and message passing without assuming gold/topic-entity access." The two rows of check 2 (webqsp 0.9215 at 2,120 candidates vs 94.9 % at 1,430 entities; metaqa 0.9876 at 2,017 vs 99.0 % at 498) and webqsp's knee shortfall (0.8628 < 0.90, fallback rule).

**Filed before the eval, read in it (note 2, H_MP):** Learned message passing pays where the gold that retrieval misses is reachable through paths that are discriminative inside the pool: few relation types, short chains, and a 1-3-hop neighbourhood of the seeds that is not saturated with equally connected distractors (metaqa: 9 relation types, answers at a fixed hop). It does not pay merely because the headroom is large: webqsp (7,058 relation types, 3-hop FULL pools of ~2,100 nodes, any_gold_at_pool 0.9215 but all_gold_at_pool 0.688) is the case where the beyond-retrieval gold exists in the pool and the arms cannot separate it, and learned propagation adds little over the fixed aggregations B / C / D / GCS that every arm already receives. In the eval, not here: the per-dataset delta_MP with its seed-paired bootstrap (measurement.delta_MP), the substrate ablation of the selected GAT (STRUCT / NER / KNN against FULL), and recall sliced by the gold's BFS distance from the seed set (the topology columns dist0_* .. dist3_* / distunreached_* of the feature contract already hold that bucket per candidate, so the slice needs no new computation and no new model). If webqsp's delta_MP stays near zero while its 2-3-hop gold stays unranked by every arm, H_MP is supported and the paper says so; if the increment concentrates on the far gold wherever it exists, the original "beyond-retrieval evidence" reading is the better one and is used instead. Either way the sentence enters the paper after the eval, with the slice beside it.

**The historical MetaQA and WebQSP rows are not comparable with these.** Under regime 2 the neighbouring entity names and relation phrases sit inside the node's own text, so a question can reach the answer node by text match alone -- Dense/SPLADE were doing part of the graph retrieval before any structural channel ran, and the graph-versus-semantic isolation was compromised. That is why the old MetaQA looked much better, and why regime 1 is the cleaner setting for the question this phase asks even though its scores are lower. Named in the M3A budget already (docs/M3A_WEBQSP_ENCODE_BUDGET.md: name_plus_facts moves relational information into the semantic channel and delta_MP stops isolating what it is meant to isolate). No number from the graph.pt substrate (M2, S3 / S4 / A3, the old MetaQA and WebQSP rows) is comparable with a number from this phase without this caveat beside it. The report prints it wherever the two meet.

How the local facts re-enter here without message passing: The semantic channel stays name-only. The local facts re-enter as declared, parameter-free functions of the query and the served graph, computed at inference time and fed identically to every arm: direction B (typed relations -- cos(q, e_r) over the edges between v and the seeds and over v's in-pool edges, relation IEF and diversity, edge direction), direction C (neighbour aggregation -- the prototype and the max cosine over v's neighbours' name embeddings per family, projected products in the model), direction D (seed-conditioned aggregation), and the fixed propagation GCS (a two-step, alpha 0.5, parameter-free spread of the fused retrieval score over the pool graph). None of these is learned message passing: they are fixed aggregations of the KG that a NAME_ONLY index cannot see. Whether this is enough to recover the old scores is one of the readings of the main table (Fixed -> QLS-U on metaqa and webqsp).

## 7. Parameters, training cost, latency, memory

| fit | parameters | epochs run | best epoch | seconds | per-epoch seconds | peak RSS GB | threads | select macro R@5 |
|---|---|---|---|---|---|---|---|---|
| `gat_no_mp_v1__H128_L2__s0` | 353,410 | 6 | 4 | 17184 | 2448 / 2478 / 2462 / 2618 / 3373 / 3804 | 8.15 | 8 | 0.7746 |
| `gat_no_mp_v1__H128_L2__s1` | 353,410 | 6 | 4 | 16179 | 3996 / 3045 / 2387 / 2296 / 2244 / 2209 | 8.83 | 8 | 0.7752 |
| `gat_no_mp_v1__H128_L2__s2` | 353,410 | 6 | 3 | 25134 | 3913 / 3874 / 4844 / 4701 / 3850 / 3950 | 11.18 | 8 | 0.7740 |
| `gat_universal_v1__H128_L2__s0` | 353,410 | 6 | 5 | 34768 | 5004 / 4712 / 5051 / 6682 / 6208 / 7110 | 9.67 | 8 | 0.8109 |
| `gat_universal_v1__H128_L2__s0__KNN` | 353,410 | 6 | 5 | 15776 | 2238 / 2247 / 2387 / 2884 / 3119 / 2901 | 10.92 | 8 | 0.7735 |
| `gat_universal_v1__H128_L2__s0__NER` | 353,410 | 6 | 3 | 24271 | 4662 / 4851 / 4955 / 4255 / 2806 / 2741 | 10.52 | 8 | 0.7718 |
| `gat_universal_v1__H128_L2__s0__STRUCT` | 353,410 | 6 | 3 | 23207 | 3719 / 3552 / 3841 / 3699 / 4416 / 3979 | 10.88 | 8 | 0.8101 |
| `gat_universal_v1__H128_L2__s1` | 353,410 | 6 | 4 | 44812 | 6378 / 11068 / 7965 / 6156 / 6554 / 6689 | 8.93 | 8 | 0.8108 |
| `gat_universal_v1__H128_L2__s2` | 353,410 | 6 | 5 | 34654 | 4097 / 3862 / 5246 / 7238 / 7238 / 6972 | 11.18 | 8 | 0.8107 |
| `gat_universal_v1__H128_L3__s0` | 387,970 | 6 | 3 | 36534 | 5134 / 5125 / 6393 / 7466 / 6813 / 5602 | 10.56 | 8 | 0.8099 |
| `gat_universal_v1__H64_L2__s0` | 259,394 | 6 | 5 | 35214 | 7319 / 6071 / 5612 / 6159 / 6709 / 3342 | 8.75 | 8 | 0.8041 |
| `gat_universal_v1__H64_L3__s0` | 268,482 | 6 | 5 | 30420 | 6179 / 4971 / 4888 / 4972 / 5216 / 4194 | 9.30 | 8 | 0.8066 |
| `qls_u_sota_v1__H128__s0` | 317,314 | 6 | 4 | 19702 | 2913 / 3516 / 2482 / 2686 / 4243 / 3861 | 8.76 | 8 | 0.7729 |
| `qls_u_sota_v1__H128__s1` | 317,314 | 6 | 5 | 16485 | 2940 / 2474 / 3204 / 2684 / 2418 / 2764 | 9.23 | 8 | 0.7725 |
| `qls_u_sota_v1__H128__s2` | 317,314 | 6 | 3 | 19828 | 2272 / 2410 / 2410 / 8846 / 2170 / 1718 | 10.82 | 8 | 0.7702 |
| `qls_u_sota_v1__H64__s0` | 249,538 | 6 | 4 | 11917 | 1588 / 1584 / 2040 / 1840 / 2166 / 2697 | 8.76 | 8 | 0.7690 |

Wall clock is CPU time on a shared machine and is not comparable with the SOTA systems' GPU-hours: the user's laptop (12 logical CPUs, 15.7 GB), shared with the user's own work throughout: a Next.js dev server (up to 7 cores while it compiles, 4-6 GB resident), a media player, browsers and the Claude app; the baseline load with nothing of this phase running was 55-75 percent and free physical memory 1.5-5.6 GB. The same computation therefore varies several-fold with the hour: the GATv2 (L 3, H 128) step on the heaviest mixed batch (19,436 candidates, 432,179 edges) took 2.2-2.4 s when the machine was quiet (per-op profile, 19:20 IST, 8-10 threads), 11.4 s in the 13:22Z timing and 22.5 s in the 14:25Z timing.

Cold per-query latency (first 500 eval queries, batch of one; compile = feature construction from the caches, embeddings and stores; pack = the batch tensors; forward = one model on the packed query) and the eval process's peak resident set. The batched column is the whole pass (compile once, every model and fixed column scored) per query, node-budgeted chunks, on the shared machine with the other lane running.

| dataset | compile p50 / p95 / p99 ms | pack p50 / p95 / p99 ms | peak RSS GB | threads | eval ms/query (batched, all models) |
|---|---|---|---|---|---|
| metaqa | 90.0 / 108.1 / 121.7 | 7.0 / 10.8 / 15.0 | 3.71 | 8 | 634.53 |
| squad | 6.6 / 9.2 / 11.0 | 0.8 / 1.6 / 2.0 | 4.29 | 8 | 21.06 |
| musique | 276.7 / 530.0 / 696.5 | 50.8 / 146.2 / 223.3 | 5.95 | 6 | 1585.82 |
| hotpotqa | 43.1 / 193.8 / 248.9 | 5.9 / 41.0 / 75.4 | 6.28 | 6 | 119.05 |
| 2wiki | 58.0 / 117.9 / 142.1 | 5.6 / 18.9 / 30.0 | 7.16 | 6 | 129.04 |
| webqsp | 292.2 / 455.0 / 613.8 | 71.9 / 149.2 / 282.2 | 5.73 | 6 | 1495.46 |

Forward pass per model, p50 / p95 / p99 ms (same 500 queries, batch of one):

| model | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| `qls_u_sota_v1__H128__s0` | 28.7 / 45.4 / 53.1 | 4.8 / 6.9 / 10.1 | 61.3 / 148.8 / 229.2 | 15.1 / 62.9 / 73.6 | 12.7 / 28.1 / 40.8 | 55.9 / 112.8 / 217.3 |
| `gat_universal_v1__H128_L2__s0` | 53.2 / 85.9 / 97.8 | 8.4 / 11.8 / 16.1 | 178.1 / 334.8 / 466.3 | 27.6 / 101.8 / 118.7 | 22.3 / 46.2 / 56.6 | 111.3 / 243.2 / 412.4 |
| `gat_no_mp_v1__H128_L2__s0` | 37.7 / 64.9 / 80.5 | 7.3 / 9.9 / 12.5 | 85.1 / 214.1 / 264.8 | 22.2 / 89.2 / 108.6 | 19.1 / 38.5 / 49.3 | 76.6 / 145.1 / 255.6 |
| `qls_u_sota_v1__H128__s1` | 28.4 / 46.8 / 58.2 | 4.7 / 7.1 / 10.1 | 59.8 / 153.1 / 226.9 | 14.3 / 59.0 / 69.1 | 12.2 / 24.5 / 31.7 | 53.0 / 129.6 / 216.6 |
| `gat_universal_v1__H128_L2__s1` | 52.8 / 89.2 / 100.9 | 8.5 / 11.7 / 16.4 | 183.8 / 345.3 / 481.1 | 27.8 / 102.7 / 125.9 | 22.1 / 43.6 / 56.2 | 109.4 / 271.6 / 395.4 |
| `gat_no_mp_v1__H128_L2__s1` | 37.7 / 63.1 / 75.0 | 7.2 / 10.5 / 14.8 | 84.7 / 222.3 / 277.4 | 22.9 / 90.9 / 104.1 | 18.8 / 35.2 / 51.2 | 75.3 / 152.4 / 251.1 |
| `qls_u_sota_v1__H128__s2` | 28.5 / 45.4 / 58.3 | 4.8 / 6.9 / 10.4 | 60.5 / 190.0 / 230.7 | 14.2 / 59.6 / 70.4 | 12.3 / 24.4 / 36.3 | 53.5 / 107.1 / 221.6 |
| `gat_universal_v1__H128_L2__s2` | 53.0 / 85.2 / 100.3 | 8.6 / 12.4 / 18.3 | 181.2 / 347.4 / 470.1 | 26.7 / 104.8 / 118.8 | 22.6 / 42.1 / 53.5 | 113.3 / 247.3 / 320.8 |
| `gat_no_mp_v1__H128_L2__s2` | 37.5 / 62.2 / 78.4 | 7.3 / 10.3 / 14.7 | 84.6 / 236.3 / 281.4 | 22.1 / 91.2 / 108.8 | 19.1 / 37.6 / 52.1 | 78.1 / 146.1 / 245.1 |
| `gat_universal_v1__H128_L2__s0__STRUCT` | 51.4 / 81.6 / 100.3 | 7.8 / 10.7 / 16.4 | 162.0 / 317.2 / 441.0 | 25.6 / 98.4 / 115.0 | 21.5 / 43.2 / 49.1 | 99.6 / 191.4 / 343.7 |
| `gat_universal_v1__H128_L2__s0__NER` | 38.7 / 62.5 / 78.7 | 7.7 / 11.6 / 15.1 | 120.2 / 268.2 / 386.6 | 24.3 / 92.3 / 113.2 | 20.7 / 40.6 / 51.6 | 87.0 / 161.9 / 272.2 |
| `gat_universal_v1__H128_L2__s0__KNN` | 39.3 / 64.3 / 78.1 | 7.6 / 10.9 / 13.2 | 90.2 / 200.1 / 271.3 | 23.9 / 94.9 / 106.9 | 19.9 / 38.4 / 50.0 | 84.4 / 227.3 / 276.4 |

## 8. The MRR audit

Every published MRR was recomputed from the stored per-query first-gold rank by an independent function (agreement to 1e-12) and hit@1 ≤ MRR ≤ 1 was asserted on every row of every scorer; `tests/test_m3b_models.py` holds the metric function against a brute-force implementation.

| dataset | scorers audited | max |recomputed − stored| | hit@1 ≤ MRR ≤ 1 |
|---|---|---|---|
| metaqa | 15 | 0.0e+00 | yes |
| squad | 15 | 0.0e+00 | yes |
| musique | 15 | 0.0e+00 | yes |
| hotpotqa | 15 | 0.0e+00 | yes |
| 2wiki | 15 | 0.0e+00 | yes |
| webqsp | 15 | 0.0e+00 | yes |

## 9. Reading

The registered question asks how much effectiveness remains attributable specifically to learned message passing once candidate exposure and inference-time graph information are matched. Per dataset, on recall@5 (seed 0 paired over eval queries; the seed column is GAT − GAT-NO-MP seed by seed):

- **metaqa** (NOT_READ): δ_MP +0.147 [+0.144, +0.150] (excludes zero); by seed +0.147, +0.146, +0.147. GAT − QLS-U +0.151 [+0.148, +0.154]. On the other metrics, δ_MP: hit@1 +0.220 [+0.215, +0.225]; mrr +0.176 [+0.173, +0.180]; full_coverage@5 +0.144 [+0.141, +0.148]. Single-family message edges (one fit each): STRUCT +0.144, NER -0.004, KNN +0.003; the best single family (STRUCT) gives +0.144 against the full GAT's +0.147.
- **squad** (CONTROL): δ_MP -0.011 [-0.015, -0.008] (excludes zero); by seed -0.011, +0.007, -0.011. GAT − QLS-U -0.005 [-0.009, -0.001]. On the other metrics, δ_MP: hit@1 -0.016 [-0.021, -0.011]; mrr -0.014 [-0.017, -0.011]; full_coverage@5 -0.011 [-0.015, -0.008]. Single-family message edges (one fit each): STRUCT +0.011, NER +0.010, KNN -0.008; the best single family (STRUCT) gives +0.011 against the full GAT's -0.011.
- **musique** (READ): δ_MP +0.018 [+0.008, +0.027] (excludes zero); by seed +0.018, +0.026, +0.028. GAT − QLS-U +0.029 [+0.020, +0.038]. On the other metrics, δ_MP: hit@1 +0.010 [-0.007, +0.026]; mrr -0.000 [-0.011, +0.010]; full_coverage@5 +0.049 [+0.036, +0.063]. Single-family message edges (one fit each): STRUCT +0.023, NER +0.018, KNN +0.005; the best single family (STRUCT) gives +0.023 against the full GAT's +0.018.
- **hotpotqa** (READ): δ_MP +0.030 [+0.026, +0.035] (excludes zero); by seed +0.030, +0.026, +0.013. GAT − QLS-U +0.026 [+0.021, +0.030]. On the other metrics, δ_MP: hit@1 -0.010 [-0.018, -0.003]; mrr -0.008 [-0.013, -0.004]; full_coverage@5 +0.068 [+0.060, +0.077]. Single-family message edges (one fit each): STRUCT +0.032, NER +0.012, KNN +0.003; the best single family (STRUCT) gives +0.032 against the full GAT's +0.030.
- **2wiki** (READ): δ_MP +0.052 [+0.048, +0.055] (excludes zero); by seed +0.052, +0.044, +0.038. GAT − QLS-U +0.047 [+0.044, +0.051]. On the other metrics, δ_MP: hit@1 -0.021 [-0.026, -0.015]; mrr -0.015 [-0.018, -0.012]; full_coverage@5 +0.126 [+0.118, +0.133]. Single-family message edges (one fit each): STRUCT +0.055, NER +0.021, KNN +0.011; the best single family (STRUCT) gives +0.055 against the full GAT's +0.052.
- **webqsp** (NOT_READ): δ_MP +0.028 [+0.014, +0.044] (excludes zero); by seed +0.028, +0.023, +0.042. GAT − QLS-U +0.025 [+0.011, +0.041]. On the other metrics, δ_MP: hit@1 +0.051 [+0.029, +0.074]; mrr +0.040 [+0.025, +0.056]; full_coverage@5 +0.023 [+0.007, +0.040]. Single-family message edges (one fit each): STRUCT +0.022, NER -0.013, KNN -0.012; the best single family (STRUCT) gives +0.022 against the full GAT's +0.028.

Read against the three outcomes filed in advance (δ_MP ≈ 0 everywhere; small on the passage graphs but substantial on the KB graphs; substantial everywhere): on the passage graphs inside the published band (musique, hotpotqa, 2wiki) δ_MP on recall@5 spans +0.018 to +0.052; on the KB graphs (metaqa, webqsp) it spans +0.028 to +0.147, but metaqa and webqsp are NOT_READ against the published band (section 6): the universal GAT sits below the band by more than the measured exposure shortfall, so that δ_MP is a measurement on an arm weaker than the published systems and is not read as the answer for the cell; squad is the control with no graph exposure in its pool (retrieval-only), where message passing over the pool graph is measured as a cost, not a regime result.

Where the increment sits: δ_MP on full_coverage@5 (every gold of the query in the top 5) is metaqa +0.144, squad -0.011, musique +0.049, hotpotqa +0.068, 2wiki +0.126, webqsp +0.023; on hit@1 it is metaqa +0.220, squad -0.016, musique +0.010, hotpotqa -0.010, 2wiki -0.021, webqsp +0.051 (intervals in section 4). On musique, hotpotqa and 2wiki message passing adds coverage of the further golds of a multi-gold query while the top rank is not lifted (its hit@1 interval does not exclude zero on the positive side); on metaqa and webqsp it lifts the top rank as well; on squad it lowers both.

**H_MP (note 2, filed before the eval).** It predicted that learned message passing pays where the gold retrieval misses is reachable through paths that are discriminative inside the pool (metaqa: 9 relation types, answers at a fixed hop) and adds little where the pool is large and the relation vocabulary wide (webqsp). Measured: metaqa δ_MP +0.147 [+0.144, +0.150] against webqsp +0.028 [+0.014, +0.044] (pool all-gold exposure 0.894 vs 0.688; any-gold 0.988 vs 0.921). The ordering H_MP predicted is the ordering measured; both KB cells carry the band verdict above.

**Where the select carve was not distribution-matched with the eval population** (section 4b): musique (select carve: 0.945 of queries hold a fit gold; eval: 0.000; fixed base shift -0.064, arms -0.257, -0.242, -0.240). The three arms lose alike, so the controlled differences survive the shift; the absolute learned lift on such a dataset is population-specific and the select-carve numbers of section 3 are read as model selection only.

Nothing here reads a negative or small δ_MP as a statement that message passing is unnecessary, and nothing reads a positive one beyond its interval; the NOT_READ cells are the calibration verdicts of section 6, not results.

## 10. Run record

Rendered 2026-09-19T10:57:19Z by `scripts/m3b_report.py` from the sidecars below (sha256 of every output a number above cites; the sidecars are gitignored, the record is committed in the declaration).

| file | sha256 |
|---|---|
| `outputs/m3b/base_score.json` | `b32b4c549c47c08c12aec645a72eb0f5b18fb2fbe1993ac34ba55394c35d422a` |
| `outputs/m3b/carves.json` | `ac481f1145cdb3b553eb9f75e2e3a48f515b8659a1f086fb6ff9759bb69dc2a1` |
| `outputs/m3b/contract/CONTRACT.json` | `d7d206c268762c074aec31e0d776b8b167fb2928a7dc9711df536c33082316f3` |
| `outputs/m3b/eval/2wiki.json` | `d9056d9c14368d10387819c03d621acbd9435976abf3c069de9537f955f6601f` |
| `outputs/m3b/eval/2wiki.npz` | `d2196c2f7769512071f1a9592504be6e1bdeff9729f1b2bcfae45f1ebdd76313` |
| `outputs/m3b/eval/2wiki_query_ids.json` | `d439a33c22f39186f8e54f596aeb2bcb83a5d27745a0dc9285e8990f16923cd2` |
| `outputs/m3b/eval/hotpotqa.json` | `00a0737d28175865a911fbdf8fad8b9d98d2cbda4ab13b72980f9251f6ce6739` |
| `outputs/m3b/eval/hotpotqa.npz` | `38927e915433fb3aea7946d92676eb1281128693aff31e37f257976a2caa23dd` |
| `outputs/m3b/eval/hotpotqa_query_ids.json` | `4d171f8d6c9b67d7a7edd01fd14dcc842d8ed42a0054b7ea32e87e728327141a` |
| `outputs/m3b/eval/metaqa.json` | `0c49fa89194642c279b1cf7ed9661c2d479e129b0c9528ed7d24ba259f91499e` |
| `outputs/m3b/eval/metaqa.npz` | `a4739cf66a8d0eb4a751abb0e9480305986eceb3876aca7662a05c7f0dfb97dd` |
| `outputs/m3b/eval/metaqa__shard0of4.json` | `b56140349b75fd9cbf2aa0467801fe78c1efa063c9cbe319e70160fb974e0cc6` |
| `outputs/m3b/eval/metaqa__shard0of4.npz` | `ea074b2194c4c4af4596467aa7dc78008fb63fb97435a7152d4c05b419a121a9` |
| `outputs/m3b/eval/metaqa__shard0of4_query_ids.json` | `18666dd2d9471151f3f83e10c8eaec95d91ef4734e36b344fc31fcea24d3e88a` |
| `outputs/m3b/eval/metaqa__shard1of4.json` | `ac7cb05a0be63fe8f1a69c77f55d4440676f9ac582529fc768817729d6703b77` |
| `outputs/m3b/eval/metaqa__shard1of4.npz` | `cd9c78cedfba30d544715a304e5c883295b1ece6bab02c8b099b0bdd85e16010` |
| `outputs/m3b/eval/metaqa__shard1of4_query_ids.json` | `7bad769398839bd5c9c1a6be0984489c6e5280f39b7732d9776685e47768f78b` |
| `outputs/m3b/eval/metaqa__shard2of4.json` | `9f265ccb0ac40337b7f0ede09fbed60a0df4f27a0407d41a3c5109ac663597fc` |
| `outputs/m3b/eval/metaqa__shard2of4.npz` | `8c4a4aa2ac34a277691ddddbda76fe7f20321b5b097cfafc208a0a6cc27a10f9` |
| `outputs/m3b/eval/metaqa__shard2of4_query_ids.json` | `70aae27b1fb90c435be556a7d9331f52d9c975b05be9728892bb2369b8c0f7e9` |
| `outputs/m3b/eval/metaqa__shard3of4.json` | `083069a32eaebc92509e3f0b28e25e2efb0d743ce697055fc7393ff2fee9dfd3` |
| `outputs/m3b/eval/metaqa__shard3of4.npz` | `091ba918716f3f41fc75a6ef8226651d8071a9f56677f6ce29c24a691c00ae91` |
| `outputs/m3b/eval/metaqa__shard3of4_query_ids.json` | `3388ee12a0352eb01577de90c0e64210a63ebf641414cb86fcc8086fea22bc35` |
| `outputs/m3b/eval/metaqa_query_ids.json` | `b491d9e90e786c1d356d2a24142ec1b810e657efdfadff85a44ff1b489618692` |
| `outputs/m3b/eval/musique.json` | `e69e211d4deff1381c33942d8acc91bb535aa91e5bdb7d328b077c9338230a61` |
| `outputs/m3b/eval/musique.npz` | `554b7a20605b6d965f3af62477103dc7c66e05079b716d781d2fa3d3408ee06c` |
| `outputs/m3b/eval/musique_query_ids.json` | `4412638ce9a31e412545fb750130f843f5abbdc8e52849cfe6dea840acbd1024` |
| `outputs/m3b/eval/squad.json` | `1d1791cf71a445672834bd74fef719185c432e9c47f371faa0e641ceecec78ae` |
| `outputs/m3b/eval/squad.npz` | `c14fe820295dc18374ef8f0d5775f2d422af830bc9a579f527736929aeef8385` |
| `outputs/m3b/eval/squad_query_ids.json` | `3ce32ec30d8920c8e0be25c2d53e325b29e4d16a5b16f7dfc9f564eb86d8a93c` |
| `outputs/m3b/eval/webqsp.json` | `4fe1f0417678a5c5dea76da2680dfb3a267c3f3523255885c6d990df7d078cf0` |
| `outputs/m3b/eval/webqsp.npz` | `8fd546c740a59d65e2d5952f563b07682a51b78fff873a87fcca29edfac446a9` |
| `outputs/m3b/eval/webqsp_query_ids.json` | `6292e45188980c4dd91606656f405d9435932fa9368fa575019590d68143ef19` |
| `outputs/m3b/feature_screen.json` | `f5c87507bbf36e3f073e4f72c009706c0b9ecf70264905cd20d7251c95b00f0c` |
| `outputs/m3b/models/gat_no_mp_v1__H128_L2__s0.json` | `f1c17437c8dc82d1f508505c8ab74ebed908b3c9d6c07954b1bafe9820e5ef37` |
| `outputs/m3b/models/gat_no_mp_v1__H128_L2__s1.json` | `6c838cf8a768a496d2ac926bf26aa046904b534a9164086af557c3a582b1e795` |
| `outputs/m3b/models/gat_no_mp_v1__H128_L2__s2.json` | `7f341765caf7486e751e815d69a51801ce363579654b9bfad5cf49da9d8540c2` |
| `outputs/m3b/models/gat_universal_v1__H128_L2__s0.json` | `99ac7215d311a392b9a4a1f60b6d91abf4e6a32fda48b0dd6d7c9b252b3d6847` |
| `outputs/m3b/models/gat_universal_v1__H128_L2__s0__KNN.json` | `d9730f037b5789f3876a545f3760aaccc36f4b77089b4735da7c2e87cc9bc9d6` |
| `outputs/m3b/models/gat_universal_v1__H128_L2__s0__NER.json` | `de1988eb847d694cfe1ad1cc07f0ae3476ec7f13bbb86e0b8d4f82478855dd87` |
| `outputs/m3b/models/gat_universal_v1__H128_L2__s0__STRUCT.json` | `13a5f1c1b481fff212a8bb0268e6ffdcd71b535b4346f009f44c5c9b28a91b47` |
| `outputs/m3b/models/gat_universal_v1__H128_L2__s1.json` | `d03de19aa7156e990a115649fb9045e64c4725be44200e5fc71958e4bd2a7a78` |
| `outputs/m3b/models/gat_universal_v1__H128_L2__s2.json` | `0d489a72d569e151b70b484c6dbfbbbeba79ebfab4f0428f03bccb39aa0c1802` |
| `outputs/m3b/models/gat_universal_v1__H128_L3__s0.json` | `a7e2f8cb22b4f95ffa3d7aaa8a471d72df7fbe50b08227b7938f125d4d517b10` |
| `outputs/m3b/models/gat_universal_v1__H64_L2__s0.json` | `26d350927734d778c99f9aed43859e717f42ab9fe56e098030e48df2e5742937` |
| `outputs/m3b/models/gat_universal_v1__H64_L3__s0.json` | `26edd4006871f2040a9c217d60385d9b6b80669899e5e51af2a2372c2b918973` |
| `outputs/m3b/models/qls_u_sota_v1__H128__s0.json` | `e4e0c6b9b18e9075f6a7ea63e63cce5ea72e1535e3407568d36c9aae23c6a465` |
| `outputs/m3b/models/qls_u_sota_v1__H128__s1.json` | `6674f75609763bc9f5bc8aa83c02a1da1a9b5ffa28be27b6546abc1ddb131e09` |
| `outputs/m3b/models/qls_u_sota_v1__H128__s2.json` | `31599eb4fa7ee715d916b582a785df0f05fa0dced1d52ee7527c95e784d3ae95` |
| `outputs/m3b/models/qls_u_sota_v1__H64__s0.json` | `96e988eb6f3f85b80aafc34b543ca9a5fe23a9a886296d3f693ccb09081d8a1e` |
| `outputs/m3b/population_overlap.json` | `2f7be185f9a5142734605e4324c2aa8ddac6f499198d3e62230f27de4f63abc7` |
| `outputs/m3b/selection.json` | `d2455bb5cadb5b9c60e2b02e36be64e58e2a87a0bf2caf36a9f57a821148ba45` |

