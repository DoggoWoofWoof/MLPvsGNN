# Universal-v2 pilot — the trio under UNIVERSAL_V2_CORE_CONTRACT

**Registered question of this phase.** Can depth-preserving compiled reasoning and a query-conditioned semantic-relation operator close the MetaQA weakness of the current universal models without sacrificing their behaviour on text retrieval?

**Registered question of the paper (unchanged).** After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?

Declaration: `configs/universal_v2.yaml` (status `PILOT_GATE_READ`; gate filed as `pilot_gate_record_2026_09_21`). A pilot on the three M3B eval populations (metaqa dev, 2wiki dev, squad dev; never a test split), each split by the filed rule into `V2_GATE` (the gate half, read once for the selected GNN and the selected twin) and `V2_HELD_CONFIRMATION` (the reported column, read once by `scripts/universal_v2_report.py --stage held`). The held half is not previously unseen data and not an independent test set: M3B scored these populations whole, and the M3B numbers on both halves are on file. Select-carve numbers select and are never results. The substrate is the read-only served package pinned by its freeze record; M3B is pinned byte-for-byte and never re-run.

**At a glance.** **GNN_GATE FAIL** / **TWIN_GATE FAIL** / overall **PILOT_FAILED** (amendment 2 family_status_vocabulary; a one-family pass is that family's pass, never a pass of the proposed universal pair). Terminal state **PILOT_FAILED** (amendment 3 terminal_state_vocabulary: GNN_GATE GATE_FAIL, TWIN_GATE GATE_FAIL; a confirmation of one family is never a confirmation of the pair). gnn `u_gnn_v2_ef` (GNN_GATE FAIL): gate **FAIL** on V2_GATE (seed 0); on V2_HELD_CONFIRMATION metaqa/hit@1/all 0.916 vs 0.884; metaqa/hit@1/3hop 0.862 vs 0.817; 2wiki/recall@5/all 0.893 vs 0.876; 2wiki/full_coverage@5/all 0.765 vs 0.745; squad/recall@5/all 0.904 vs 0.900. twin `u_mlp_v2_mix` (TWIN_GATE FAIL): gate **FAIL** on V2_GATE (seed 0); on V2_HELD_CONFIRMATION metaqa/hit@1/all 0.785 vs 0.702; 2wiki/recall@5/all 0.853 vs 0.858; squad/recall@5/all 0.912 vs 0.900. The held-half cells are confirmatory readings of the gate's arms, not a gate; nothing here is a statement about message passing (the paper's question is answered by M3B) or a comparison to the published systems (their exposure is named in section 10).

## 1. The frozen contract (`contract_frozen_2026_09_19`)

164 raw columns (the M3B 78 first, then the 86 v2 columns of `information_contract_v2`: 73 depth-basis and 13 ordered relation-path columns, pinned by `amendment_2_2026_09_19`) were screened on the fit carves only, under the M3B rule verbatim; 129 survive as `UNIVERSAL_V2_CORE_CONTRACT` (sha256 `8d1da88b14dfdee1…`). The M3B 78 are protected earlier members and always survive; the trio-only statistics that would have dropped one of them are recorded, not applied.

| hashed at contract_frozen | sha256 |
|---|---|
| raw_contract_sha256 | `41a3f66792de250ee424ea7370b0736fb6035fff5b834be318be1377086270ba` |
| screen_rule_sha256 | `109632d8e6b5dc75836671c39f8e62222463a60c0803a74d02b3d11b186dbcdb` |
| surviving_columns_sha256 | `8d1da88b14dfdee13dd32ee6a05f6d9733d14a9899d5e011c41b0cce53000bf9` |
| six_caches_combined_sha256 | `0042c258bd7bb7bfd5684521acb4e97bd80984d82a6cb70a5b18574c9c7c0aa1` |

| dropped column | reason |
|---|---|
| `seeds_at_h1_STRUCT` | later_member_of_duplicate_pair |
| `seedmass_h1_STRUCT` | later_member_of_duplicate_pair |
| `paths_h1_STRUCT` | later_member_of_duplicate_pair |
| `support_h1_STRUCT` | later_member_of_duplicate_pair |
| `has_h1_STRUCT` | later_member_of_duplicate_pair |
| `ring_qmax_h1_STRUCT` | later_member_of_duplicate_pair |
| `ring_n_h1_STRUCT` | later_member_of_duplicate_pair |
| `seedproto_h1_STRUCT` | later_member_of_duplicate_pair |
| `seedmass_h2_STRUCT` | later_member_of_duplicate_pair |
| `paths_h2_STRUCT` | later_member_of_duplicate_pair |
| `branch_h2_STRUCT` | later_member_of_duplicate_pair |
| `seeds_at_h1_FULL` | later_member_of_duplicate_pair |
| `seedmass_h1_FULL` | later_member_of_duplicate_pair |
| `paths_h1_FULL` | later_member_of_duplicate_pair |
| `support_h1_FULL` | later_member_of_duplicate_pair |
| `has_h1_FULL` | later_member_of_duplicate_pair |
| `ring_n_h1_FULL` | later_member_of_duplicate_pair |
| `seedproto_h1_FULL` | later_member_of_duplicate_pair |
| `seedmass_h2_FULL` | later_member_of_duplicate_pair |
| `paths_h2_FULL` | later_member_of_duplicate_pair |
| `branch_h2_FULL` | later_member_of_duplicate_pair |
| `has_h2_FULL` | later_member_of_duplicate_pair |
| `ring_n_h2_FULL` | later_member_of_duplicate_pair |
| `qsupport_h1` | later_member_of_duplicate_pair |
| `typed_walks_h1` | later_member_of_duplicate_pair |
| `relpath_max_h1` | later_member_of_duplicate_pair |
| `relpath_mean_h1` | later_member_of_duplicate_pair |
| `relpath_min_h1` | later_member_of_duplicate_pair |
| `typed_walks_h2` | later_member_of_duplicate_pair |
| `relpath_max_h2` | later_member_of_duplicate_pair |
| `relpath_mean_h2` | later_member_of_duplicate_pair |
| `relpath_min_h2` | later_member_of_duplicate_pair |
| `opath_h2_q1` | later_member_of_duplicate_pair |
| `opath_h2_q2` | later_member_of_duplicate_pair |
| `opath_h2_dir2` | later_member_of_duplicate_pair |

M3B columns the trio statistics alone would have dropped (protected, kept): `seed_component_NER` (later_member_of_duplicate_pair), `component_size_NER` (later_member_of_duplicate_pair), `cos_q_proto_ner` (later_member_of_duplicate_pair), `cohesion_ner` (later_member_of_duplicate_pair), `has_nbr_ner` (later_member_of_duplicate_pair).

Duplicate pairs among the new columns (earlier member kept): `dist1_STRUCT`~`seeds_at_h1_STRUCT` (0.984), `dist1_STRUCT`~`seedmass_h1_STRUCT` (0.984), `dist1_STRUCT`~`paths_h1_STRUCT` (0.984), `dist1_STRUCT`~`support_h1_STRUCT` (0.985), `dist1_STRUCT`~`has_h1_STRUCT` (0.985), `dist1_STRUCT`~`seedproto_h1_STRUCT` (0.983), `deg_pool_STRUCT`~`ring_n_h1_STRUCT` (1.000), `walks2_STRUCT`~`paths_h2_STRUCT` (1.000), `walks2_STRUCT`~`branch_h2_STRUCT` (0.991), `distunreached_NER`~`seed_component_NER` (0.995), `deg_pool_NER`~`component_size_NER` (1.000), `deg_pool_NER`~`cos_q_proto_ner` (0.996), `deg_pool_NER`~`cohesion_ner` (0.992), `deg_pool_NER`~`has_nbr_ner` (0.998), `component_size_NER`~`cos_q_proto_ner` (0.996), `component_size_NER`~`cohesion_ner` (0.991), `component_size_NER`~`has_nbr_ner` (0.998), `seeds_1hop_FULL`~`seeds_at_h1_FULL` (1.000), `seeds_1hop_FULL`~`seedmass_h1_FULL` (1.000), `seeds_1hop_FULL`~`paths_h1_FULL` (1.000), `seeds_1hop_FULL`~`support_h1_FULL` (0.999), `seeds_1hop_FULL`~`has_h1_FULL` (1.000), `seeds_1hop_FULL`~`seedproto_h1_FULL` (0.999), `deg_pool_FULL`~`ring_n_h1_FULL` (1.000), `walks2_FULL`~`paths_h2_FULL` (1.000), `walks2_FULL`~`branch_h2_FULL` (0.990), `max_q_nbr_structural`~`ring_qmax_h1_STRUCT` (1.000), `cos_q_proto_ner`~`cohesion_ner` (0.990), `cos_q_proto_ner`~`has_nbr_ner` (0.996), `cohesion_ner`~`has_nbr_ner` (0.996), `has_seed_h2`~`has_h2_FULL` (0.989), `relmax_seed`~`qsupport_h1` (0.999), `relmax_seed`~`typed_walks_h1` (0.999), `relmax_seed`~`relpath_max_h1` (1.000), `relmax_seed`~`relpath_mean_h1` (1.000), `relmax_seed`~`relpath_min_h1` (1.000), `relchain2_max`~`relpath_max_h2` (0.986), `relchain2_max`~`relpath_mean_h2` (0.986), `relchain2_max`~`relpath_min_h2` (0.981), `relchain2_max`~`opath_h2_q1` (0.984), `relchain2_max`~`opath_h2_q2` (0.982), `seeds_at_h1_STRUCT`~`seedmass_h1_STRUCT` (1.000), `seeds_at_h1_STRUCT`~`paths_h1_STRUCT` (1.000), `seeds_at_h1_STRUCT`~`support_h1_STRUCT` (0.999), `seeds_at_h1_STRUCT`~`has_h1_STRUCT` (1.000), `seeds_at_h1_STRUCT`~`seedproto_h1_STRUCT` (0.999), `seedmass_h1_STRUCT`~`paths_h1_STRUCT` (1.000), `seedmass_h1_STRUCT`~`support_h1_STRUCT` (0.999), `seedmass_h1_STRUCT`~`has_h1_STRUCT` (0.999), `seedmass_h1_STRUCT`~`seedproto_h1_STRUCT` (0.999), `paths_h1_STRUCT`~`support_h1_STRUCT` (0.999), `paths_h1_STRUCT`~`has_h1_STRUCT` (1.000), `paths_h1_STRUCT`~`seedproto_h1_STRUCT` (0.999), `support_h1_STRUCT`~`has_h1_STRUCT` (0.999), `support_h1_STRUCT`~`seedproto_h1_STRUCT` (0.999), `has_h1_STRUCT`~`seedproto_h1_STRUCT` (0.999), `seeds_at_h2_STRUCT`~`seedmass_h2_STRUCT` (0.991), `paths_h2_STRUCT`~`branch_h2_STRUCT` (0.991), `ring_n_h2_STRUCT`~`ring_n_h2_FULL` (0.988), `seeds_at_h1_FULL`~`seedmass_h1_FULL` (1.000), `seeds_at_h1_FULL`~`paths_h1_FULL` (1.000), `seeds_at_h1_FULL`~`support_h1_FULL` (0.999), `seeds_at_h1_FULL`~`has_h1_FULL` (1.000), `seeds_at_h1_FULL`~`seedproto_h1_FULL` (0.999), `seedmass_h1_FULL`~`paths_h1_FULL` (1.000), `seedmass_h1_FULL`~`support_h1_FULL` (0.999), `seedmass_h1_FULL`~`has_h1_FULL` (0.999), `seedmass_h1_FULL`~`seedproto_h1_FULL` (0.999), `paths_h1_FULL`~`support_h1_FULL` (0.999), `paths_h1_FULL`~`has_h1_FULL` (1.000), `paths_h1_FULL`~`seedproto_h1_FULL` (0.999), `support_h1_FULL`~`has_h1_FULL` (0.999), `support_h1_FULL`~`seedproto_h1_FULL` (0.998), `has_h1_FULL`~`seedproto_h1_FULL` (0.999), `seeds_at_h2_FULL`~`seedmass_h2_FULL` (0.989), `paths_h2_FULL`~`branch_h2_FULL` (0.990), `qsupport_h1`~`typed_walks_h1` (1.000), `qsupport_h1`~`relpath_max_h1` (1.000), `qsupport_h1`~`relpath_mean_h1` (1.000), `qsupport_h1`~`relpath_min_h1` (1.000), `typed_walks_h1`~`relpath_max_h1` (1.000), `typed_walks_h1`~`relpath_mean_h1` (1.000), `typed_walks_h1`~`relpath_min_h1` (1.000), `relpath_max_h1`~`relpath_mean_h1` (1.000), `relpath_max_h1`~`relpath_min_h1` (1.000), `relpath_mean_h1`~`relpath_min_h1` (1.000), `qsupport_h2`~`typed_walks_h2` (0.983), `qsupport_h2`~`relpath_max_h2` (0.988), `qsupport_h2`~`relpath_mean_h2` (0.987), `qsupport_h2`~`relpath_min_h2` (0.983), `qsupport_h2`~`opath_h2_q1` (0.989), `qsupport_h2`~`opath_h2_q2` (0.982), `typed_walks_h2`~`relpath_max_h2` (0.983), `typed_walks_h2`~`relpath_min_h2` (0.982), `typed_walks_h2`~`opath_h2_q1` (0.982), `typed_walks_h2`~`opath_h2_q2` (0.982), `typed_walks_h2`~`opath_h2_adj12` (0.981), `relpath_max_h2`~`relpath_mean_h2` (0.999), `relpath_max_h2`~`relpath_min_h2` (0.996), `relpath_max_h2`~`opath_h2_q1` (0.998), `relpath_max_h2`~`opath_h2_q2` (0.996), `relpath_mean_h2`~`relpath_min_h2` (0.994), `relpath_mean_h2`~`opath_h2_q1` (0.997), `relpath_mean_h2`~`opath_h2_q2` (0.994), `relpath_min_h2`~`opath_h2_q1` (0.991), `relpath_min_h2`~`opath_h2_q2` (0.998), `relpath_min_h2`~`opath_h2_adj12` (0.982), `opath_h2_q1`~`opath_h2_q2` (0.989), `opath_h2_dir1`~`opath_h2_dir2` (0.989).

| dataset | carve | queries | rows | candidates (mean) | compile ms/query | queries / s | wall s | peak RSS GB | cache GB | seeds added / query |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | fit | 5,960 | 12,021,002 | 2016.9 | 423.02 | 2.36 | 2521 | 2.73 | 4.70 | 0.02 |
| metaqa | select | 1,497 | 3,015,964 | 2014.7 | 401.77 | 2.49 | 602 | 2.73 | 1.18 | 0.02 |
| 2wiki | fit | 5,928 | 628,354 | 106.0 | 30.87 | 32.40 | 183 | 3.92 | 0.26 | 0.01 |
| 2wiki | select | 1,496 | 159,930 | 106.9 | 26.38 | 37.91 | 40 | 4.21 | 0.07 | 0.01 |
| squad | fit | 5,856 | 292,994 | 50.0 | 8.66 | 115.51 | 51 | 4.21 | 0.13 | 0.03 |
| squad | select | 1,498 | 74,946 | 50.0 | 8.67 | 115.40 | 13 | 4.21 | 0.03 | 0.03 |

K_REL = 4 relation-text slots per structural message edge: how often a structural pair carries more stored relations than the slots hold, over every query of every carve (amendment 2 `k_rel_truncation_report_required`; untyped datasets have no structural relation table and report zero pairs).

| dataset | carve | hop | queries | typed queries | pairs | typed entries | pairs truncated | fraction of pairs | fraction of queries with any | max relations / pair | pairs beyond K_REL (histogram) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | fit | 1hop | 1,740 | 1,740 | 15,374,663 | 15,898,587 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | fit | 2hop | 2,153 | 2,153 | 20,484,925 | 21,063,269 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | fit | 3hop | 2,067 | 2,067 | 18,667,278 | 19,317,732 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | fit | all | 5,960 | 5,960 | 54,526,866 | 56,279,588 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | select | 1hop | 437 | 437 | 3,897,451 | 4,027,377 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | select | 2hop | 541 | 541 | 5,212,023 | 5,353,325 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | select | 3hop | 519 | 519 | 4,693,728 | 4,855,915 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| metaqa | select | all | 1,497 | 1,497 | 13,803,202 | 14,236,617 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| 2wiki | fit | all | 5,928 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0 | 0 |
| 2wiki | select | all | 1,496 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0 | 0 |
| squad | fit | all | 5,856 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0 | 0 |
| squad | select | all | 1,498 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0 | 0 |

Typed STRUCT relation-path availability (rows with a walk of length t / gold rows / queries with any such row) and the ordered channel (best walks at t = 2, 3: rows with a walk / fraction with an inverse step / fraction composing two different relations):

| dataset | carve | hop | h1 rows / gold / queries | h2 | h3 | ordered h2 rows / inverse / heterogeneous | ordered h3 |
|---|---|---|---|---|---|---|---|
| metaqa | fit | 1hop | 0.026 / 0.954 / 1.000 | 0.275 / 0.163 / 1.000 | 0.800 / 1.000 / 1.000 | 962,995 / 0.994 / 0.190 | 2,802,768 / 1.000 / 0.858 |
| metaqa | fit | 2hop | 0.024 / 0.034 / 1.000 | 0.238 / 0.972 / 1.000 | 0.838 / 0.586 / 1.000 | 1,025,646 / 0.997 / 0.334 | 3,613,402 / 1.000 / 0.856 |
| metaqa | fit | 3hop | 0.029 / 0.079 / 1.000 | 0.308 / 0.142 / 1.000 | 0.820 / 0.998 / 1.000 | 1,297,388 / 0.995 / 0.242 | 3,448,681 / 1.000 / 0.862 |
| metaqa | fit | all | 0.026 / 0.138 / 1.000 | 0.273 / 0.424 / 1.000 | 0.821 / 0.859 / 1.000 | 3,286,029 / 0.996 / 0.256 | 9,864,851 / 1.000 / 0.859 |
| metaqa | select | 1hop | 0.025 / 0.956 / 1.000 | 0.269 / 0.170 / 1.000 | 0.803 / 1.000 / 1.000 | 236,739 / 0.995 / 0.209 | 707,136 / 1.000 / 0.857 |
| metaqa | select | 2hop | 0.023 / 0.043 / 1.000 | 0.233 / 0.972 / 1.000 | 0.840 / 0.570 / 1.000 | 251,242 / 0.997 / 0.349 | 907,396 / 1.000 / 0.852 |
| metaqa | select | 3hop | 0.030 / 0.077 / 1.000 | 0.312 / 0.160 / 1.000 | 0.818 / 0.999 / 1.000 | 329,175 / 0.996 / 0.244 | 862,898 / 1.000 / 0.860 |
| metaqa | select | all | 0.026 / 0.142 / 1.000 | 0.271 / 0.406 / 1.000 | 0.821 / 0.869 / 1.000 | 817,156 / 0.996 / 0.266 | 2,477,430 / 1.000 / 0.856 |
| 2wiki | fit | all | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0.000 / 0.000 | 0 / 0.000 / 0.000 |
| 2wiki | select | all | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0.000 / 0.000 | 0 / 0.000 / 0.000 |
| squad | fit | all | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0.000 / 0.000 | 0 / 0.000 / 0.000 |
| squad | select | all | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0.000 / 0.000 | 0 / 0.000 / 0.000 |

## 2. Cost: the timing run, the fits, the eval pass

Timing run (`timing.json`, before any fit): one epoch of `u_gnn_v2_ef` = 4866 s at 8 threads (2000 batches of 16); fallback fires: False. Per arm on three mixed batches:

| arm | parameters | s / batch | projected minutes / epoch |
|---|---|---|---|
| `u_mlp_v2` | 330,370 | 0.79 | 26.2 |
| `u_mlp_v2_mix` | 330,955 | 0.57 | 19.0 |
| `u_gnn_v2` | 417,603 | 2.28 | 76.1 |
| `u_gnn_v2_ef` | 420,932 | 2.25 | 74.9 |
| `u_gnn_v2_core78` | 407,876 | 2.43 | 81.1 |
| `gat_universal_v1_trio` | 353,410 | 0.81 | 26.8 |

| fit | parameters (budget) | epochs | best | seconds | s / epoch | peak RSS GB | threads | select macro R@5 (selection only) |
|---|---|---|---|---|---|---|---|---|
| `gat_universal_v1_trio__H128__s0` | 353,410 (450,000) | 4 | 1 | 12213 | 3053 | 9.11 | 8 | 0.8719 |
| `u_gnn_v2__H128__s0` | 417,603 (450,000) | 4 | 1 | 18622 | 4656 | 8.60 | 8 | 0.8748 |
| `u_gnn_v2_core78__H128__s0` | 407,876 (450,000) | 4 | 1 | 31830 | 7957 | 8.15 | 8 | 0.8777 |
| `u_gnn_v2_ef__H128__s0` | 420,932 (450,000) | 4 | 1 | 18060 | 4515 | 8.64 | 8 | 0.8777 |
| `u_mlp_v2__H128__s0` | 330,370 (350,000) | 4 | 1 | 3924 | 981 | 8.79 | 8 | 0.8457 |
| `u_mlp_v2_mix__H128__s0` | 330,955 (350,000) | 3 | 0 | 4087 | 1362 | 9.23 | 8 | 0.8457 |

| eval record | queries | ms / query | compile p50 / p95 ms | pack p50 / p95 ms | forward p50 ms per model | peak RSS GB |
|---|---|---|---|---|---|---|
| `2wiki` | 12,576 | 83.69 | 51.8 / 240.2 | 6.1 / 30.9 | `gat_universal_v1_trio__H128__s0` 22.9; `u_gnn_v2__H128__s0` 30.8; `u_gnn_v2_core78__H128__s0` 36.2; `u_gnn_v2_ef__H128__s0` 35.0; `u_mlp_v2__H128__s0` 11.7; `u_mlp_v2_mix__H128__s0` 12.2 | 5.51 |
| `metaqa` | 39,138 | 1205.94 | 655.8 / 2133.0 | 13.2 / 82.7 | `gat_universal_v1_trio__H128__s0` 84.4; `u_gnn_v2__H128__s0` 161.8; `u_gnn_v2_core78__H128__s0` 199.9; `u_gnn_v2_ef__H128__s0` 204.2; `u_mlp_v2__H128__s0` 49.3; `u_mlp_v2_mix__H128__s0` 50.2 | 3.15 |
| `squad` | 11,873 | 94.05 | 30.6 / 58.4 | 2.5 / 5.4 | `gat_universal_v1_trio__H128__s0` 30.9; `u_gnn_v2__H128__s0` 41.8; `u_gnn_v2_core78__H128__s0` 49.2; `u_gnn_v2_ef__H128__s0` 47.1; `u_mlp_v2__H128__s0` 15.4; `u_mlp_v2_mix__H128__s0` 16.4 | 3.37 |

## 3. Selection behind the firewall

Within each family the seed-0 candidate with the highest macro select recall@5 (ties to the fewer parameters); `selection.json` was written once, before any eval population was scored, and no number of the other family or of M3B entered it. The ablation `u_gnn_v2_core78` takes the selected GNN architecture on the M3B 78 and feeds no selection; `gat_universal_v1_trio` is the control.

| family | candidate | select macro R@5 | parameters | best epoch | seconds |
|---|---|---|---|---|---|
| gnn | `u_gnn_v2` | 0.8748 | 417,603 | 1 | 18622 |
| gnn | `u_gnn_v2_ef` **(selected)** | 0.8777 | 420,932 | 1 | 18060 |
| twin | `u_mlp_v2` | 0.8457 | 330,370 | 1 | 3924 |
| twin | `u_mlp_v2_mix` **(selected)** | 0.8457 | 330,955 | 0 | 4087 |

## 4. The pilot gate (V2_GATE, seed 0, read once)

Thresholds from `amendment_1_2026_09_19.pilot_gate_amended`, filed before any weight existed; paired intervals: bootstrap over the gate-half queries, 1000 resamples, `default_rng(0)`, 95 % percentile. The gate is read for the selected GNN and the selected twin only; a non-selected candidate is reported below and cannot advance. What the gate is not: not a result on the held half (never read for the gate), not a statement about message passing (the registered question of the paper is answered by M3B), and not a comparison to the published systems (their exposure is named in the calibration table; methodological_ruling_band_rule).

**`u_gnn_v2_ef` — FAIL**

| dataset | metric | slice | queries | value | threshold | paired interval | cell |
|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,738 | 0.9175 | 0.8840 | +0.086 [+0.081, +0.090] vs `gat_universal_v1`, excludes zero: True | holds |
| metaqa | hit@1 | 3hop | 7,273 | 0.8679 | 0.8172 | +0.129 [+0.119, +0.140] vs `gat_universal_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,290 | 0.8963 | 0.8758 | — | holds |
| 2wiki | full_coverage@5 | all | 6,290 | 0.7674 | 0.7453 | — | holds |
| squad | recall@5 | all | 5,841 | 0.8988 | 0.8996 | — | FAILS |

**`u_mlp_v2_mix` — FAIL**

| dataset | metric | slice | queries | value | threshold | paired interval | cell |
|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,738 | 0.7772 | 0.7021 | +0.162 [+0.155, +0.169] vs `qls_u_sota_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,290 | 0.8545 | 0.8578 | — | FAILS |
| squad | recall@5 | all | 5,841 | 0.9105 | 0.8996 | — | holds |

Reported, not advancing (the other candidate of each family, on the same cells):

`u_gnn_v2` (would hold the cells; not the selected candidate of its family (arms.selection_behind_the_firewall)):

| dataset | metric | slice | queries | value | threshold | paired interval | cell |
|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,738 | 0.9167 | 0.8840 | +0.085 [+0.080, +0.090] vs `gat_universal_v1`, excludes zero: True | holds |
| metaqa | hit@1 | 3hop | 7,273 | 0.8633 | 0.8172 | +0.125 [+0.115, +0.135] vs `gat_universal_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,290 | 0.8948 | 0.8758 | — | holds |
| 2wiki | full_coverage@5 | all | 6,290 | 0.7653 | 0.7453 | — | holds |
| squad | recall@5 | all | 5,841 | 0.8997 | 0.8996 | — | holds |

`u_mlp_v2` (would fail the cells; not the selected candidate of its family (arms.selection_behind_the_firewall)):

| dataset | metric | slice | queries | value | threshold | paired interval | cell |
|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,738 | 0.7941 | 0.7021 | +0.179 [+0.172, +0.185] vs `qls_u_sota_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,290 | 0.8502 | 0.8578 | — | FAILS |
| squad | recall@5 | all | 5,841 | 0.8867 | 0.8996 | — | FAILS |

On a pass: seeds 1 and 2 of the passing arm are fitted; the pass is confirmed if the mean over seeds 0-2 also holds every cell; a confirmed pass makes the arm eligible for the six-dataset stage (later_stages), which still needs a dated authorization block. On a fail: the arm stops at seed 0 with its numbers in the report; no threshold is moved, no candidate is added and no fit is repeated to reach the gate. If both selected arms fail, the phase closes with STOP_FOR_REVIEW and the numbers; what to try next is a new dated amendment, filed before its fit.

## 5. Seed confirmation (V2_GATE, seeds 0-2)

No arm passed its gate; no seeds 1-2 were fitted (pilot_gate.on_fail).

Seed mean ± sd on V2_GATE for every arm with more than one seed:

| dataset | arm | seeds | recall@5 | hit@1 | full_coverage@5 | mrr |
|---|---|---|---|---|---|---|

## 6. V2_HELD_CONFIRMATION — the reported column (read once)

Read once at 2026-09-21T04:59:21Z (status `PILOT_GATE_READ`, gate record `8dcc217b08a7…`), after the gate was filed; no selection, threshold, screen or checkpoint decision saw it. Every v2 arm, the fixed scorers and the frozen M3B references on the same queries (the frozen `fixed:rrf` array equals the v2 row per query, section 10, and is listed once). The gate-half reads of the same arms are section 4 (V2_GATE, seed 0) and the gate record's summary; the counts of both halves are beside each population.

**metaqa** (19,400 held queries; 19,738 gate queries)

| scorer | recall@1 | recall@5 | recall@20 | hit@1 | mrr | full_coverage@5 |
|---|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0293 | 0.0637 | 0.1490 | 0.0436 | 0.0777 | 0.0508 |
| `fixed:qsupport_h2` | 0.0041 | 0.0295 | 0.0829 | 0.0078 | 0.0340 | 0.0218 |
| `fixed:qsupport_h3` | 0.0285 | 0.0664 | 0.1461 | 0.0401 | 0.0722 | 0.0555 |
| fixed rrf `fixed:rrf` | 0.0004 | 0.0053 | 0.0151 | 0.0008 | 0.0157 | 0.0035 |
| `fixed:support_h1_STRUCT` | 0.0204 | 0.0592 | 0.1574 | 0.0306 | 0.0698 | 0.0456 |
| `fixed:support_h2_STRUCT` | 0.0061 | 0.0349 | 0.0823 | 0.0128 | 0.0408 | 0.0240 |
| `fixed:support_h3_STRUCT` | 0.0160 | 0.0501 | 0.1443 | 0.0224 | 0.0574 | 0.0402 |
| universal GAT, trio (control) s0 `gat_universal_v1_trio__H128__s0` | 0.4623 | 0.7712 | 0.9067 | 0.8547 | 0.9024 | 0.6492 |
| U-GNN-v2 s0 `u_gnn_v2__H128__s0` | 0.4937 | 0.7918 | 0.9174 | 0.9162 | 0.9403 | 0.6749 |
| U-GNN-v2 on the M3B 78 (ablation) s0 `u_gnn_v2_core78__H128__s0` | 0.4800 | 0.7916 | 0.9191 | 0.9024 | 0.9337 | 0.6721 |
| U-GNN-v2-EF s0 `u_gnn_v2_ef__H128__s0` | 0.4942 | 0.7931 | 0.9184 | 0.9158 | 0.9411 | 0.6755 |
| U-MLP-v2 s0 `u_mlp_v2__H128__s0` | 0.4388 | 0.7310 | 0.8857 | 0.7933 | 0.8608 | 0.6044 |
| U-MLP-v2-mix s0 `u_mlp_v2_mix__H128__s0` | 0.4357 | 0.7216 | 0.8797 | 0.7848 | 0.8528 | 0.5961 |
| universal GAT (frozen M3B) `gat_universal_v1__H128_L2__s0` | — | 0.7676 | — | 0.8300 | 0.8858 | 0.6476 |
| GAT-NO-MP (frozen M3B) `gat_no_mp_v1__H128_L2__s0` | — | 0.6210 | — | 0.6069 | 0.7091 | 0.5030 |
| QLS-U (frozen M3B) `qls_u_sota_v1__H128__s0` | — | 0.6162 | — | 0.6093 | 0.7111 | 0.4984 |

**2wiki** (6,286 held queries; 6,290 gate queries)

| scorer | recall@1 | recall@5 | recall@20 | hit@1 | mrr | full_coverage@5 |
|---|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0067 | 0.0439 | 0.1796 | 0.0173 | 0.0839 | 0.0017 |
| `fixed:qsupport_h2` | 0.0067 | 0.0439 | 0.1796 | 0.0173 | 0.0839 | 0.0017 |
| `fixed:qsupport_h3` | 0.0067 | 0.0439 | 0.1796 | 0.0173 | 0.0839 | 0.0017 |
| fixed rrf `fixed:rrf` | 0.3864 | 0.6036 | 0.6897 | 0.8746 | 0.9239 | 0.2464 |
| `fixed:support_h1_STRUCT` | 0.0329 | 0.0834 | 0.1884 | 0.0861 | 0.1611 | 0.0008 |
| `fixed:support_h2_STRUCT` | 0.1763 | 0.5064 | 0.7046 | 0.4189 | 0.6054 | 0.1745 |
| `fixed:support_h3_STRUCT` | 0.0213 | 0.0705 | 0.2027 | 0.0585 | 0.1378 | 0.0010 |
| universal GAT, trio (control) s0 `gat_universal_v1_trio__H128__s0` | 0.3987 | 0.8945 | 0.9514 | 0.9036 | 0.9395 | 0.7724 |
| U-GNN-v2 s0 `u_gnn_v2__H128__s0` | 0.3923 | 0.8938 | 0.9527 | 0.8894 | 0.9332 | 0.7668 |
| U-GNN-v2 on the M3B 78 (ablation) s0 `u_gnn_v2_core78__H128__s0` | 0.3839 | 0.8991 | 0.9539 | 0.8718 | 0.9229 | 0.7806 |
| U-GNN-v2-EF s0 `u_gnn_v2_ef__H128__s0` | 0.3939 | 0.8932 | 0.9524 | 0.8947 | 0.9367 | 0.7655 |
| U-MLP-v2 s0 `u_mlp_v2__H128__s0` | 0.3877 | 0.8491 | 0.9434 | 0.8786 | 0.9270 | 0.6677 |
| U-MLP-v2-mix s0 `u_mlp_v2_mix__H128__s0` | 0.3991 | 0.8526 | 0.9479 | 0.9038 | 0.9411 | 0.6729 |
| universal GAT (frozen M3B) `gat_universal_v1__H128_L2__s0` | — | 0.8842 | — | 0.8818 | 0.9247 | 0.7574 |
| GAT-NO-MP (frozen M3B) `gat_no_mp_v1__H128_L2__s0` | — | 0.8334 | — | 0.9014 | 0.9398 | 0.6314 |
| QLS-U (frozen M3B) `qls_u_sota_v1__H128__s0` | — | 0.8363 | — | 0.8734 | 0.9233 | 0.6422 |

**squad** (6,032 held queries; 5,841 gate queries)

| scorer | recall@1 | recall@5 | recall@20 | hit@1 | mrr | full_coverage@5 |
|---|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0181 | 0.0932 | 0.3808 | 0.0181 | 0.0859 | 0.0932 |
| `fixed:qsupport_h2` | 0.0181 | 0.0932 | 0.3808 | 0.0181 | 0.0859 | 0.0932 |
| `fixed:qsupport_h3` | 0.0181 | 0.0932 | 0.3808 | 0.0181 | 0.0859 | 0.0932 |
| fixed rrf `fixed:rrf` | 0.7420 | 0.9060 | 0.9649 | 0.7420 | 0.8162 | 0.9060 |
| `fixed:support_h1_STRUCT` | 0.0088 | 0.0734 | 0.3835 | 0.0088 | 0.0739 | 0.0734 |
| `fixed:support_h2_STRUCT` | 0.1126 | 0.3105 | 0.5721 | 0.1126 | 0.2177 | 0.3105 |
| `fixed:support_h3_STRUCT` | 0.0109 | 0.0894 | 0.4209 | 0.0109 | 0.0816 | 0.0894 |
| universal GAT, trio (control) s0 `gat_universal_v1_trio__H128__s0` | 0.7522 | 0.9083 | 0.9662 | 0.7522 | 0.8234 | 0.9083 |
| U-GNN-v2 s0 `u_gnn_v2__H128__s0` | 0.7435 | 0.9029 | 0.9619 | 0.7435 | 0.8160 | 0.9029 |
| U-GNN-v2 on the M3B 78 (ablation) s0 `u_gnn_v2_core78__H128__s0` | 0.7439 | 0.9050 | 0.9634 | 0.7439 | 0.8166 | 0.9050 |
| U-GNN-v2-EF s0 `u_gnn_v2_ef__H128__s0` | 0.7427 | 0.9038 | 0.9635 | 0.7427 | 0.8170 | 0.9038 |
| U-MLP-v2 s0 `u_mlp_v2__H128__s0` | 0.7343 | 0.8922 | 0.9536 | 0.7343 | 0.8065 | 0.8922 |
| U-MLP-v2-mix s0 `u_mlp_v2_mix__H128__s0` | 0.7601 | 0.9123 | 0.9672 | 0.7601 | 0.8299 | 0.9123 |
| universal GAT (frozen M3B) `gat_universal_v1__H128_L2__s0` | — | 0.8851 | — | 0.7251 | 0.7980 | 0.8851 |
| GAT-NO-MP (frozen M3B) `gat_no_mp_v1__H128_L2__s0` | — | 0.9025 | — | 0.7401 | 0.8125 | 0.9025 |
| QLS-U (frozen M3B) `qls_u_sota_v1__H128__s0` | — | 0.8941 | — | 0.7329 | 0.8062 | 0.8941 |

The gate cells re-read on the held half for the arms the gate selected (confirmatory: the verdict is the gate's, on the gate half):

`u_gnn_v2_ef` (gate outcome FAIL):

| dataset | metric | slice | queries | value | gate threshold | paired interval | at threshold |
|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,400 | 0.9158 | 0.8840 | +0.086 [+0.081, +0.091] vs `gat_universal_v1`, excludes zero: True | holds |
| metaqa | hit@1 | 3hop | 7,001 | 0.8620 | 0.8172 | +0.126 [+0.116, +0.136] vs `gat_universal_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,286 | 0.8932 | 0.8758 | — | holds |
| 2wiki | full_coverage@5 | all | 6,286 | 0.7655 | 0.7453 | — | holds |
| squad | recall@5 | all | 6,032 | 0.9038 | 0.8996 | — | holds |

`u_mlp_v2_mix` (gate outcome FAIL):

| dataset | metric | slice | queries | value | gate threshold | paired interval | at threshold |
|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,400 | 0.7848 | 0.7021 | +0.175 [+0.169, +0.182] vs `qls_u_sota_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,286 | 0.8526 | 0.8578 | — | FAILS |
| squad | recall@5 | all | 6,032 | 0.9123 | 0.8996 | — | holds |

Paired differences on V2_HELD_CONFIRMATION (per query, bootstrap 1000 resamples, `default_rng(0)`, 95 % percentile); the same pairs on V2_GATE are in the gate record.

| pair | metaqa recall@5 / hit@1 / full_coverage@5 | 2wiki recall@5 / hit@1 / full_coverage@5 | squad recall@5 / hit@1 / full_coverage@5 |
|---|---|---|---|
| selected gnn minus selected twin | +0.071 [+0.069, +0.074] / +0.131 [+0.126, +0.136] / +0.079 [+0.075, +0.084] | +0.041 [+0.036, +0.045] / -0.009 [-0.016, -0.002] / +0.093 [+0.084, +0.102] | -0.009 [-0.012, -0.004] / -0.017 [-0.023, -0.011] / -0.009 [-0.012, -0.004] |
| selected gnn minus u gnn v2 core78 | +0.002 [+0.000, +0.003] / +0.013 [+0.009, +0.017] / +0.003 [+0.001, +0.006] | -0.006 [-0.009, -0.003] / +0.023 [+0.016, +0.030] / -0.015 [-0.022, -0.008] | -0.001 [-0.005, +0.003] / -0.001 [-0.006, +0.004] / -0.001 [-0.005, +0.003] |
| selected gnn minus gat universal v1 trio | +0.022 [+0.020, +0.024] / +0.061 [+0.057, +0.065] / +0.026 [+0.024, +0.029] | -0.001 [-0.005, +0.002] / -0.009 [-0.015, -0.003] / -0.007 [-0.014, +0.000] | -0.004 [-0.008, -0.001] / -0.009 [-0.015, -0.004] / -0.004 [-0.008, -0.001] |
| u gnn v2 core78 minus gat universal v1 trio | +0.020 [+0.019, +0.022] / +0.048 [+0.043, +0.052] / +0.023 [+0.020, +0.026] | +0.005 [+0.001, +0.008] / -0.032 [-0.039, -0.025] / +0.008 [+0.001, +0.015] | -0.003 [-0.007, +0.000] / -0.008 [-0.014, -0.003] / -0.003 [-0.007, +0.000] |
| selected gnn minus gat universal v1 frozen | +0.025 [+0.024, +0.027] / +0.086 [+0.081, +0.091] / +0.028 [+0.025, +0.031] | +0.009 [+0.005, +0.013] / +0.013 [+0.005, +0.021] / +0.008 [-0.000, +0.016] | +0.019 [+0.014, +0.024] / +0.018 [+0.010, +0.025] / +0.019 [+0.014, +0.024] |
| selected twin minus qls u sota v1 frozen | +0.105 [+0.101, +0.109] / +0.175 [+0.169, +0.182] / +0.098 [+0.093, +0.103] | +0.016 [+0.011, +0.021] / +0.030 [+0.023, +0.038] / +0.031 [+0.021, +0.040] | +0.018 [+0.013, +0.023] / +0.027 [+0.021, +0.034] / +0.018 [+0.013, +0.023] |
| selected twin minus gat no mp v1 frozen | +0.101 [+0.096, +0.104] / +0.178 [+0.170, +0.185] / +0.093 [+0.088, +0.098] | +0.019 [+0.015, +0.024] / +0.002 [-0.004, +0.009] / +0.042 [+0.032, +0.051] | +0.010 [+0.006, +0.014] / +0.020 [+0.014, +0.026] / +0.010 [+0.006, +0.014] |

## 7. Slices (measurement.slices_reported)

**metaqa by hop, hit@1 / recall@5, V2_HELD_CONFIRMATION**

| scorer | 1hop (4,971) | 2hop (7,428) | 3hop (7,001) |
|---|---|---|---|
| `fixed:qsupport_h1` | 0.1652 / 0.2326 | 0.0003 / 0.0012 | 0.0033 / 0.0101 |
| `fixed:qsupport_h2` | 0.0018 / 0.0053 | 0.0188 / 0.0729 | 0.0004 / 0.0008 |
| `fixed:qsupport_h3` | 0.1489 / 0.2332 | 0.0004 / 0.0015 | 0.0049 / 0.0168 |
| `fixed:rrf` | 0.0016 / 0.0086 | 0.0009 / 0.0069 | 0.0001 / 0.0013 |
| `fixed:support_h1_STRUCT` | 0.1131 / 0.2018 | 0.0004 / 0.0011 | 0.0040 / 0.0196 |
| `fixed:support_h2_STRUCT` | 0.0016 / 0.0047 | 0.0322 / 0.0878 | 0.0003 / 0.0004 |
| `fixed:support_h3_STRUCT` | 0.0805 / 0.1647 | 0.0004 / 0.0011 | 0.0046 / 0.0208 |
| `gat_universal_v1_trio__H128__s0` | 0.9006 / 0.9573 | 0.9019 / 0.8371 | 0.7722 / 0.5691 |
| `u_gnn_v2__H128__s0` | 0.9226 / 0.9603 | 0.9615 / 0.8537 | 0.8637 / 0.6064 |
| `u_gnn_v2_core78__H128__s0` | 0.9219 / 0.9661 | 0.9237 / 0.8498 | 0.8659 / 0.6059 |
| `u_gnn_v2_ef__H128__s0` | 0.9346 / 0.9641 | 0.9538 / 0.8539 | 0.8620 / 0.6071 |
| `u_mlp_v2__H128__s0` | 0.8980 / 0.9541 | 0.8880 / 0.8243 | 0.6185 / 0.4738 |
| `u_mlp_v2_mix__H128__s0` | 0.8843 / 0.9458 | 0.8916 / 0.8215 | 0.6008 / 0.4563 |

**recall@5 by the gold's STRUCT distance from the seeds, V2_HELD_CONFIRMATION** (dist0..dist3, unreached; queries in brackets)

metaqa:

| scorer | dist0 (540) | dist1 (10,005) | dist2 (7,077) | dist3 (1,528) | unreached (1) |
|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0472 | 0.1210 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:qsupport_h2` | 0.1415 | 0.0003 | 0.0698 | 0.0000 | 0.0000 |
| `fixed:qsupport_h3` | 0.0401 | 0.1254 | 0.0000 | 0.0080 | 0.0000 |
| `fixed:rrf` | 0.1457 | 0.0012 | 0.0016 | 0.0007 | 0.0000 |
| `fixed:support_h1_STRUCT` | 0.0583 | 0.1117 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:support_h2_STRUCT` | 0.1253 | 0.0007 | 0.0853 | 0.0000 | 0.0000 |
| `fixed:support_h3_STRUCT` | 0.0434 | 0.0934 | 0.0002 | 0.0082 | 0.0000 |
| `gat_universal_v1_trio__H128__s0` | 0.5379 | 0.7971 | 0.8193 | 0.5871 | 0.0370 |
| `u_gnn_v2__H128__s0` | 0.5660 | 0.8100 | 0.8388 | 0.6641 | 0.0000 |
| `u_gnn_v2_core78__H128__s0` | 0.5717 | 0.8123 | 0.8355 | 0.6597 | 0.0370 |
| `u_gnn_v2_ef__H128__s0` | 0.5710 | 0.8127 | 0.8383 | 0.6636 | 0.0000 |
| `u_mlp_v2__H128__s0` | 0.4784 | 0.7536 | 0.7928 | 0.5062 | 0.0000 |
| `u_mlp_v2_mix__H128__s0` | 0.4077 | 0.7426 | 0.7869 | 0.5104 | 0.0000 |

2wiki:

| scorer | dist0 (6,243) | dist1 (13) | dist2 (11) | unreached (17) |
|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0440 | 0.0769 | 0.0000 | 0.0000 |
| `fixed:qsupport_h2` | 0.0440 | 0.0769 | 0.0000 | 0.0000 |
| `fixed:qsupport_h3` | 0.0440 | 0.0769 | 0.0000 | 0.0000 |
| `fixed:rrf` | 0.6066 | 0.1154 | 0.0909 | 0.2647 |
| `fixed:support_h1_STRUCT` | 0.0839 | 0.0385 | 0.0000 | 0.0000 |
| `fixed:support_h2_STRUCT` | 0.5098 | 0.0385 | 0.0000 | 0.0000 |
| `fixed:support_h3_STRUCT` | 0.0708 | 0.0769 | 0.0000 | 0.0000 |
| `gat_universal_v1_trio__H128__s0` | 0.8992 | 0.3077 | 0.1364 | 0.2059 |
| `u_gnn_v2__H128__s0` | 0.8980 | 0.3846 | 0.1364 | 0.3235 |
| `u_gnn_v2_core78__H128__s0` | 0.9034 | 0.3846 | 0.0909 | 0.3235 |
| `u_gnn_v2_ef__H128__s0` | 0.8977 | 0.3846 | 0.1364 | 0.2353 |
| `u_mlp_v2__H128__s0` | 0.8532 | 0.4231 | 0.1818 | 0.2059 |
| `u_mlp_v2_mix__H128__s0` | 0.8570 | 0.4615 | 0.0455 | 0.1471 |

squad:

| scorer | dist0 (5,645) | dist1 (71) | dist2 (21) | dist3 (1) | unreached (177) |
|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0957 | 0.0986 | 0.0952 | 0.0000 | 0.0734 |
| `fixed:qsupport_h2` | 0.0957 | 0.0986 | 0.0952 | 0.0000 | 0.0734 |
| `fixed:qsupport_h3` | 0.0957 | 0.0986 | 0.0952 | 0.0000 | 0.0734 |
| `fixed:rrf` | 0.9637 | 0.0986 | 0.1905 | 0.0000 | 0.0791 |
| `fixed:support_h1_STRUCT` | 0.0716 | 0.4225 | 0.0000 | 0.0000 | 0.0508 |
| `fixed:support_h2_STRUCT` | 0.3272 | 0.1690 | 0.3333 | 0.0000 | 0.0395 |
| `fixed:support_h3_STRUCT` | 0.0889 | 0.3944 | 0.0000 | 1.0000 | 0.0452 |
| `gat_universal_v1_trio__H128__s0` | 0.9649 | 0.1127 | 0.1429 | 0.0000 | 0.1186 |
| `u_gnn_v2__H128__s0` | 0.9575 | 0.1690 | 0.0952 | 0.0000 | 0.1525 |
| `u_gnn_v2_core78__H128__s0` | 0.9607 | 0.1268 | 0.0476 | 0.0000 | 0.1469 |
| `u_gnn_v2_ef__H128__s0` | 0.9589 | 0.1408 | 0.0952 | 0.0000 | 0.1525 |
| `u_mlp_v2__H128__s0` | 0.9474 | 0.1831 | 0.0476 | 0.0000 | 0.1130 |
| `u_mlp_v2_mix__H128__s0` | 0.9672 | 0.1127 | 0.0000 | 0.0000 | 0.1977 |

**multi-gold queries (2+ in-pool golds), full_coverage@5 / recall@5, V2_HELD_CONFIRMATION**

| scorer | metaqa (11,502) | 2wiki (6,002) | squad (0) |
|---|---|---|---|
| `fixed:qsupport_h1` | 0.0163 / 0.0380 | 0.0018 / 0.0450 | — |
| `fixed:qsupport_h2` | 0.0045 / 0.0177 | 0.0018 / 0.0450 | — |
| `fixed:qsupport_h3` | 0.0198 / 0.0377 | 0.0018 / 0.0450 | — |
| `fixed:rrf` | 0.0003 / 0.0032 | 0.2581 / 0.6100 | — |
| `fixed:support_h1_STRUCT` | 0.0118 / 0.0342 | 0.0008 / 0.0872 | — |
| `fixed:support_h2_STRUCT` | 0.0049 / 0.0234 | 0.1828 / 0.5150 | — |
| `fixed:support_h3_STRUCT` | 0.0130 / 0.0291 | 0.0010 / 0.0734 | — |
| `gat_universal_v1_trio__H128__s0` | 0.4576 / 0.6600 | 0.8089 / 0.9152 | — |
| `u_gnn_v2__H128__s0` | 0.4963 / 0.6898 | 0.8031 / 0.9139 | — |
| `u_gnn_v2_core78__H128__s0` | 0.4916 / 0.6895 | 0.8176 / 0.9199 | — |
| `u_gnn_v2_ef__H128__s0` | 0.4962 / 0.6910 | 0.8017 / 0.9136 | — |
| `u_mlp_v2__H128__s0` | 0.3945 / 0.6052 | 0.6993 / 0.8680 | — |
| `u_mlp_v2_mix__H128__s0` | 0.3867 / 0.5954 | 0.7048 / 0.8714 | — |

**metaqa by hop, hit@1 / recall@5, V2_GATE**

| scorer | 1hop (5,021) | 2hop (7,444) | 3hop (7,273) |
|---|---|---|---|
| `fixed:qsupport_h1` | 0.1637 / 0.2360 | 0.0001 / 0.0005 | 0.0025 / 0.0110 |
| `fixed:qsupport_h2` | 0.0010 / 0.0037 | 0.0195 / 0.0785 | 0.0008 / 0.0010 |
| `fixed:qsupport_h3` | 0.1492 / 0.2385 | 0.0005 / 0.0008 | 0.0049 / 0.0182 |
| `fixed:rrf` | 0.0016 / 0.0063 | 0.0004 / 0.0069 | 0.0000 / 0.0013 |
| `fixed:support_h1_STRUCT` | 0.1155 / 0.2029 | 0.0000 / 0.0005 | 0.0030 / 0.0203 |
| `fixed:support_h2_STRUCT` | 0.0014 / 0.0037 | 0.0341 / 0.0901 | 0.0001 / 0.0008 |
| `fixed:support_h3_STRUCT` | 0.0840 / 0.1766 | 0.0003 / 0.0005 | 0.0045 / 0.0203 |
| `gat_universal_v1_trio__H128__s0` | 0.9010 / 0.9582 | 0.8991 / 0.8281 | 0.7804 / 0.5574 |
| `u_gnn_v2__H128__s0` | 0.9319 / 0.9620 | 0.9586 / 0.8444 | 0.8633 / 0.5971 |
| `u_gnn_v2_core78__H128__s0` | 0.9307 / 0.9655 | 0.9262 / 0.8412 | 0.8708 / 0.5978 |
| `u_gnn_v2_ef__H128__s0` | 0.9428 / 0.9675 | 0.9490 / 0.8438 | 0.8679 / 0.5983 |
| `u_mlp_v2__H128__s0` | 0.9018 / 0.9552 | 0.8846 / 0.8100 | 0.6271 / 0.4664 |
| `u_mlp_v2_mix__H128__s0` | 0.8781 / 0.9468 | 0.8800 / 0.8103 | 0.6022 / 0.4484 |

**recall@5 by the gold's STRUCT distance from the seeds, V2_GATE** (dist0..dist3, unreached; queries in brackets)

metaqa:

| scorer | dist0 (582) | dist1 (10,236) | dist2 (7,094) | dist3 (1,588) | unreached (1) |
|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0543 | 0.1208 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:qsupport_h2` | 0.1252 | 0.0003 | 0.0755 | 0.0000 | 0.0000 |
| `fixed:qsupport_h3` | 0.0510 | 0.1259 | 0.0000 | 0.0112 | 0.0000 |
| `fixed:rrf` | 0.1230 | 0.0008 | 0.0018 | 0.0000 | 0.0000 |
| `fixed:support_h1_STRUCT` | 0.0567 | 0.1111 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:support_h2_STRUCT` | 0.1197 | 0.0006 | 0.0872 | 0.0000 | 0.0000 |
| `fixed:support_h3_STRUCT` | 0.0478 | 0.0976 | 0.0000 | 0.0071 | 0.0000 |
| `gat_universal_v1_trio__H128__s0` | 0.5747 | 0.7910 | 0.8022 | 0.5712 | 0.0000 |
| `u_gnn_v2__H128__s0` | 0.5908 | 0.8051 | 0.8225 | 0.6542 | 0.0000 |
| `u_gnn_v2_core78__H128__s0` | 0.6108 | 0.8062 | 0.8198 | 0.6514 | 0.0000 |
| `u_gnn_v2_ef__H128__s0` | 0.6063 | 0.8074 | 0.8214 | 0.6584 | 0.0000 |
| `u_mlp_v2__H128__s0` | 0.5198 | 0.7450 | 0.7722 | 0.5110 | 0.0000 |
| `u_mlp_v2_mix__H128__s0` | 0.4646 | 0.7342 | 0.7680 | 0.5121 | 0.0000 |

2wiki:

| scorer | dist0 (6,267) | dist1 (8) | dist2 (4) | unreached (9) |
|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0425 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:qsupport_h2` | 0.0425 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:qsupport_h3` | 0.0425 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:rrf` | 0.6140 | 0.0625 | 0.1250 | 0.2500 |
| `fixed:support_h1_STRUCT` | 0.0837 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:support_h2_STRUCT` | 0.5132 | 0.0000 | 0.0000 | 0.0000 |
| `fixed:support_h3_STRUCT` | 0.0717 | 0.0000 | 0.0000 | 0.0000 |
| `gat_universal_v1_trio__H128__s0` | 0.8966 | 0.3750 | 0.0000 | 0.4167 |
| `u_gnn_v2__H128__s0` | 0.8970 | 0.3750 | 0.1250 | 0.3333 |
| `u_gnn_v2_core78__H128__s0` | 0.9002 | 0.3750 | 0.0000 | 0.3611 |
| `u_gnn_v2_ef__H128__s0` | 0.8986 | 0.3750 | 0.1250 | 0.3333 |
| `u_mlp_v2__H128__s0` | 0.8528 | 0.1250 | 0.1250 | 0.2222 |
| `u_mlp_v2_mix__H128__s0` | 0.8569 | 0.3750 | 0.0000 | 0.1944 |

squad:

| scorer | dist0 (5,415) | dist1 (62) | dist2 (18) | dist3 (2) | unreached (221) |
|---|---|---|---|---|---|
| `fixed:qsupport_h1` | 0.0992 | 0.0806 | 0.0000 | 0.5000 | 0.0995 |
| `fixed:qsupport_h2` | 0.0992 | 0.0806 | 0.0000 | 0.5000 | 0.0995 |
| `fixed:qsupport_h3` | 0.0992 | 0.0806 | 0.0000 | 0.5000 | 0.0995 |
| `fixed:rrf` | 0.9666 | 0.1774 | 0.2222 | 0.0000 | 0.1584 |
| `fixed:support_h1_STRUCT` | 0.0659 | 0.3548 | 0.0000 | 0.5000 | 0.0543 |
| `fixed:support_h2_STRUCT` | 0.3102 | 0.0806 | 0.1111 | 0.0000 | 0.0498 |
| `fixed:support_h3_STRUCT` | 0.0827 | 0.3065 | 0.0000 | 0.5000 | 0.0543 |
| `gat_universal_v1_trio__H128__s0` | 0.9673 | 0.1613 | 0.0556 | 0.0000 | 0.1267 |
| `u_gnn_v2__H128__s0` | 0.9592 | 0.1613 | 0.1111 | 0.0000 | 0.2217 |
| `u_gnn_v2_core78__H128__s0` | 0.9610 | 0.1452 | 0.1111 | 0.0000 | 0.1629 |
| `u_gnn_v2_ef__H128__s0` | 0.9607 | 0.1613 | 0.1111 | 0.0000 | 0.1629 |
| `u_mlp_v2__H128__s0` | 0.9487 | 0.1290 | 0.0556 | 0.5000 | 0.1448 |
| `u_mlp_v2_mix__H128__s0` | 0.9705 | 0.1774 | 0.0556 | 0.0000 | 0.2308 |

**multi-gold queries (2+ in-pool golds), full_coverage@5 / recall@5, V2_GATE**

| scorer | metaqa (11,836) | 2wiki (6,020) | squad (0) |
|---|---|---|---|
| `fixed:qsupport_h1` | 0.0149 / 0.0366 | 0.0027 / 0.0436 | — |
| `fixed:qsupport_h2` | 0.0054 / 0.0182 | 0.0027 / 0.0436 | — |
| `fixed:qsupport_h3` | 0.0176 / 0.0355 | 0.0027 / 0.0436 | — |
| `fixed:rrf` | 0.0000 / 0.0040 | 0.2726 / 0.6187 | — |
| `fixed:support_h1_STRUCT` | 0.0118 / 0.0336 | 0.0012 / 0.0869 | — |
| `fixed:support_h2_STRUCT` | 0.0061 / 0.0230 | 0.1890 / 0.5193 | — |
| `fixed:support_h3_STRUCT` | 0.0121 / 0.0289 | 0.0012 / 0.0745 | — |
| `gat_universal_v1_trio__H128__s0` | 0.4393 / 0.6478 | 0.8010 / 0.9133 | — |
| `u_gnn_v2__H128__s0` | 0.4801 / 0.6801 | 0.7997 / 0.9138 | — |
| `u_gnn_v2_core78__H128__s0` | 0.4770 / 0.6793 | 0.8096 / 0.9173 | — |
| `u_gnn_v2_ef__H128__s0` | 0.4817 / 0.6816 | 0.8018 / 0.9154 | — |
| `u_mlp_v2__H128__s0` | 0.3794 / 0.5946 | 0.6987 / 0.8675 | — |
| `u_mlp_v2_mix__H128__s0` | 0.3698 / 0.5837 | 0.7055 / 0.8717 | — |

## 8. Mechanism readouts (measurement.mechanism_readouts)

`delta_ratio` = mean |delta_s| / mean |base_z| over the pool; `top1_changed` = fraction of queries whose top-1 leaves the fixed base; `gate_step{t}` = the update gate g per step (mean, q25 / q50 / q75), `gate2_step{t}` the evidence gate, `block_gate{b}` the mix twin's block gates.

**V2_HELD_CONFIRMATION**

| dataset | scorer | delta_ratio | top1_changed | gates (mean, q25 / q50 / q75) |
|---|---|---|---|---|
| metaqa | `gat_universal_v1_trio__H128__s0` | 30.2685 | 0.9994 | — |
| metaqa | `u_gnn_v2__H128__s0` | 22.5316 | 0.9995 | gate_step1 0.332 (0.27 / 0.33 / 0.39); gate_step2 0.390 (0.30 / 0.39 / 0.47); gate_step3 0.452 (0.35 / 0.45 / 0.55) |
| metaqa | `u_gnn_v2_core78__H128__s0` | 20.8803 | 0.9991 | gate2_step1 0.947 (0.94 / 0.95 / 0.96); gate2_step2 0.817 (0.79 / 0.83 / 0.85); gate2_step3 0.970 (0.96 / 0.97 / 0.98); gate_step1 0.542 (0.47 / 0.55 / 0.62); gate_step2 0.513 (0.44 / 0.52 / 0.58); gate_step3 0.523 (0.44 / 0.53 / 0.60) |
| metaqa | `u_gnn_v2_ef__H128__s0` | 19.8913 | 0.9991 | gate2_step1 0.713 (0.65 / 0.72 / 0.78); gate2_step2 0.362 (0.31 / 0.36 / 0.41); gate2_step3 0.618 (0.50 / 0.63 / 0.74); gate_step1 0.272 (0.21 / 0.26 / 0.33); gate_step2 0.305 (0.23 / 0.30 / 0.38); gate_step3 0.329 (0.24 / 0.32 / 0.41) |
| metaqa | `u_mlp_v2__H128__s0` | 67.5790 | 0.9994 | — |
| metaqa | `u_mlp_v2_mix__H128__s0` | 42.2147 | 0.9998 | block_gate0 0.514 (0.46 / 0.51 / 0.59); block_gate1 0.597 (0.48 / 0.65 / 0.72); block_gate2 0.504 (0.44 / 0.53 / 0.57); block_gate3 0.464 (0.30 / 0.39 / 0.63); block_gate4 0.581 (0.47 / 0.64 / 0.70); block_gate5 0.444 (0.42 / 0.45 / 0.48); block_gate6 0.500 (0.50 / 0.50 / 0.50); block_gate7 0.557 (0.50 / 0.56 / 0.60); block_gate8 0.573 (0.47 / 0.60 / 0.68) |
| 2wiki | `gat_universal_v1_trio__H128__s0` | 9.5552 | 0.3291 | — |
| 2wiki | `u_gnn_v2__H128__s0` | 9.7207 | 0.4640 | gate_step1 0.361 (0.28 / 0.36 / 0.45); gate_step2 0.323 (0.24 / 0.32 / 0.40); gate_step3 0.313 (0.24 / 0.31 / 0.39) |
| 2wiki | `u_gnn_v2_core78__H128__s0` | 8.9676 | 0.4846 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.439 (0.32 / 0.46 / 0.55); gate_step2 0.304 (0.24 / 0.30 / 0.36); gate_step3 0.264 (0.21 / 0.26 / 0.32) |
| 2wiki | `u_gnn_v2_ef__H128__s0` | 8.2995 | 0.4636 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.334 (0.25 / 0.33 / 0.41); gate_step2 0.265 (0.20 / 0.26 / 0.33); gate_step3 0.236 (0.18 / 0.23 / 0.29) |
| 2wiki | `u_mlp_v2__H128__s0` | 12.8463 | 0.3738 | — |
| 2wiki | `u_mlp_v2_mix__H128__s0` | 7.1877 | 0.2701 | block_gate0 0.394 (0.35 / 0.38 / 0.43); block_gate1 0.297 (0.26 / 0.29 / 0.33); block_gate2 0.410 (0.36 / 0.41 / 0.46); block_gate3 0.734 (0.68 / 0.76 / 0.81); block_gate4 0.307 (0.27 / 0.30 / 0.33); block_gate5 0.439 (0.41 / 0.44 / 0.47); block_gate6 0.500 (0.50 / 0.50 / 0.50); block_gate7 0.513 (0.48 / 0.51 / 0.54); block_gate8 0.502 (0.44 / 0.50 / 0.57) |
| squad | `gat_universal_v1_trio__H128__s0` | 9.0474 | 0.1232 | — |
| squad | `u_gnn_v2__H128__s0` | 9.8231 | 0.1369 | gate_step1 0.176 (0.12 / 0.16 / 0.22); gate_step2 0.167 (0.11 / 0.15 / 0.21); gate_step3 0.161 (0.11 / 0.15 / 0.20) |
| squad | `u_gnn_v2_core78__H128__s0` | 9.3606 | 0.1386 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.331 (0.25 / 0.32 / 0.41); gate_step2 0.212 (0.17 / 0.21 / 0.25); gate_step3 0.170 (0.13 / 0.16 / 0.20) |
| squad | `u_gnn_v2_ef__H128__s0` | 8.0895 | 0.1356 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.131 (0.08 / 0.12 / 0.16); gate_step2 0.117 (0.08 / 0.11 / 0.15); gate_step3 0.107 (0.07 / 0.10 / 0.13) |
| squad | `u_mlp_v2__H128__s0` | 14.6084 | 0.1588 | — |
| squad | `u_mlp_v2_mix__H128__s0` | 6.0760 | 0.1102 | block_gate0 0.445 (0.41 / 0.44 / 0.47); block_gate1 0.262 (0.22 / 0.25 / 0.29); block_gate2 0.335 (0.31 / 0.33 / 0.35); block_gate3 0.690 (0.63 / 0.71 / 0.77); block_gate4 0.279 (0.24 / 0.26 / 0.31); block_gate5 0.372 (0.35 / 0.37 / 0.39); block_gate6 0.500 (0.50 / 0.50 / 0.50); block_gate7 0.482 (0.45 / 0.48 / 0.51); block_gate8 0.419 (0.38 / 0.41 / 0.45) |

**V2_GATE**

| dataset | scorer | delta_ratio | top1_changed | gates (mean, q25 / q50 / q75) |
|---|---|---|---|---|
| metaqa | `gat_universal_v1_trio__H128__s0` | 30.2304 | 0.9993 | — |
| metaqa | `u_gnn_v2__H128__s0` | 22.5026 | 0.9998 | gate_step1 0.332 (0.27 / 0.33 / 0.39); gate_step2 0.391 (0.31 / 0.39 / 0.47); gate_step3 0.452 (0.35 / 0.46 / 0.55) |
| metaqa | `u_gnn_v2_core78__H128__s0` | 20.8546 | 0.9994 | gate2_step1 0.947 (0.94 / 0.95 / 0.96); gate2_step2 0.817 (0.79 / 0.83 / 0.85); gate2_step3 0.970 (0.96 / 0.97 / 0.98); gate_step1 0.543 (0.48 / 0.55 / 0.62); gate_step2 0.515 (0.45 / 0.52 / 0.59); gate_step3 0.525 (0.45 / 0.53 / 0.60) |
| metaqa | `u_gnn_v2_ef__H128__s0` | 19.8587 | 0.9990 | gate2_step1 0.713 (0.65 / 0.72 / 0.78); gate2_step2 0.362 (0.31 / 0.36 / 0.41); gate2_step3 0.619 (0.50 / 0.63 / 0.74); gate_step1 0.273 (0.21 / 0.26 / 0.33); gate_step2 0.307 (0.23 / 0.30 / 0.38); gate_step3 0.330 (0.24 / 0.32 / 0.41) |
| metaqa | `u_mlp_v2__H128__s0` | 67.3993 | 0.9993 | — |
| metaqa | `u_mlp_v2_mix__H128__s0` | 42.0919 | 0.9999 | block_gate0 0.513 (0.46 / 0.51 / 0.59); block_gate1 0.596 (0.48 / 0.65 / 0.72); block_gate2 0.504 (0.44 / 0.53 / 0.57); block_gate3 0.462 (0.30 / 0.39 / 0.63); block_gate4 0.580 (0.47 / 0.63 / 0.70); block_gate5 0.444 (0.42 / 0.45 / 0.48); block_gate6 0.500 (0.50 / 0.50 / 0.50); block_gate7 0.557 (0.51 / 0.56 / 0.60); block_gate8 0.574 (0.48 / 0.60 / 0.68) |
| 2wiki | `gat_universal_v1_trio__H128__s0` | 9.5552 | 0.3293 | — |
| 2wiki | `u_gnn_v2__H128__s0` | 9.7043 | 0.4523 | gate_step1 0.358 (0.27 / 0.36 / 0.44); gate_step2 0.321 (0.24 / 0.32 / 0.39); gate_step3 0.311 (0.23 / 0.30 / 0.38) |
| 2wiki | `u_gnn_v2_core78__H128__s0` | 8.9632 | 0.4776 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.436 (0.32 / 0.46 / 0.54); gate_step2 0.303 (0.24 / 0.30 / 0.36); gate_step3 0.263 (0.21 / 0.26 / 0.31) |
| 2wiki | `u_gnn_v2_ef__H128__s0` | 8.2816 | 0.4493 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.332 (0.25 / 0.33 / 0.41); gate_step2 0.263 (0.20 / 0.26 / 0.32); gate_step3 0.235 (0.18 / 0.23 / 0.29) |
| 2wiki | `u_mlp_v2__H128__s0` | 12.7972 | 0.3700 | — |
| 2wiki | `u_mlp_v2_mix__H128__s0` | 7.1523 | 0.2653 | block_gate0 0.395 (0.35 / 0.38 / 0.43); block_gate1 0.296 (0.26 / 0.29 / 0.33); block_gate2 0.407 (0.36 / 0.40 / 0.45); block_gate3 0.732 (0.68 / 0.76 / 0.81); block_gate4 0.307 (0.27 / 0.30 / 0.33); block_gate5 0.437 (0.41 / 0.44 / 0.47); block_gate6 0.500 (0.50 / 0.50 / 0.50); block_gate7 0.513 (0.48 / 0.51 / 0.54); block_gate8 0.499 (0.44 / 0.49 / 0.56) |
| squad | `gat_universal_v1_trio__H128__s0` | 9.0298 | 0.1395 | — |
| squad | `u_gnn_v2__H128__s0` | 9.7715 | 0.1548 | gate_step1 0.177 (0.11 / 0.16 / 0.22); gate_step2 0.168 (0.11 / 0.15 / 0.21); gate_step3 0.162 (0.10 / 0.15 / 0.20) |
| squad | `u_gnn_v2_core78__H128__s0` | 9.3164 | 0.1623 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.328 (0.24 / 0.32 / 0.41); gate_step2 0.211 (0.17 / 0.21 / 0.25); gate_step3 0.168 (0.13 / 0.16 / 0.20) |
| squad | `u_gnn_v2_ef__H128__s0` | 8.0459 | 0.1489 | gate2_step1 0.000 (0.00 / 0.00 / 0.00); gate2_step2 0.000 (0.00 / 0.00 / 0.00); gate2_step3 0.000 (0.00 / 0.00 / 0.00); gate_step1 0.130 (0.08 / 0.12 / 0.16); gate_step2 0.117 (0.08 / 0.10 / 0.15); gate_step3 0.106 (0.07 / 0.10 / 0.13) |
| squad | `u_mlp_v2__H128__s0` | 14.5190 | 0.1775 | — |
| squad | `u_mlp_v2_mix__H128__s0` | 6.0219 | 0.1192 | block_gate0 0.443 (0.41 / 0.44 / 0.47); block_gate1 0.262 (0.22 / 0.25 / 0.29); block_gate2 0.335 (0.31 / 0.33 / 0.35); block_gate3 0.689 (0.63 / 0.71 / 0.77); block_gate4 0.280 (0.23 / 0.27 / 0.31); block_gate5 0.373 (0.35 / 0.37 / 0.39); block_gate6 0.500 (0.50 / 0.50 / 0.50); block_gate7 0.482 (0.45 / 0.48 / 0.51); block_gate8 0.420 (0.38 / 0.41 / 0.46) |

## 9. Calibration against the published systems (diagnostic; exposure attached)

The READ / NOT_READ band rule of M3B (the operationalisation of configs/m3a_compilation.yaml#universal_gnn_calibration_rule .when_delta_mp_is_not_read as lowest published number minus the measured exposure shortfall) is diagnostic. Its precise operational form was written on 2026-09-19 at 13:36 IST, mid-evaluation, after four M3B records existed (M3B run record, sections_beyond_the_declared_reporting). It is never described as pre-registered, in this phase or in the paper. Families: GFM-RAG / G-reasoner (universal graph foundation rankers); GNN-RAG / ReaRev / NuTrea (KB, assigned topic entities); GraphER / GeAR (passage graphs, the closest comparator to the twin is the parameter-free GraphER arm); the fixed propagation baselines. Rule: every comparison is under retrieved exposure unless a labelled oracle-entry lane is declared and run; none is declared here.

| dataset | published (exposure) | selected GNN `u_gnn_v2_ef__H128__s0` on V2_HELD_CONFIRMATION | diagnostic band check |
|---|---|---|---|
| metaqa | NuTrea Hit@1 1-hop 97.4 / 2-hop 99.99 / 3-hop 98.89; ReaRev 3-hop 98.9 (KB exposure) | hit@1 1hop 0.935 / 2hop 0.954 / 3hop 0.862 (all 0.916) | NOT_READ: 1hop 0.935 vs 0.974 (below by 0.039; exposure shortfall 0.002); 2hop 0.954 vs 1.000 (below by 0.046; exposure shortfall 0.002); 3hop 0.862 vs 0.989 (below by 0.127; exposure shortfall 0.002) |
| 2wiki | GraphER PR@5 43.8 / 44.1 / 42.5; PR@10 51.1 / 53.0 / 51.1 | R@5 0.893 / R@10 0.934 | READ: recall@5 0.893 vs published 0.425-0.441 |
| squad | no graph-retrieval SOTA in the archaeology; the retrieval ceiling row stands in | R@5 0.904 | CONTROL: no published graph-retrieval number; the fixed rrf row stands in |

| method | graph entry point | avg graph / pool | any-answer exposure | retrieval prior retained? | ranking metric |
|---|---|---|---|---|---|
| ReaRev (EMNLP Findings 2022) | provided topic entities + PageRank-Nibble top-m (He et al. 2021) | WebQSP 1,429.8 entities (m = 2,000); MetaQA-3 497.9 (m = 500) | WebQSP 94.9 %; MetaQA-3 99.0 % (Table 6) | no -- no dense / sparse retriever in the pipeline; the question enters through the entity | Hits@1 / F1 over answer entities |
| GNN-RAG (2024) | linked entities + PageRank-Nibble top-2,000 (same preprocessing family; ReaRev is its GNN) | WebQSP 1,429.8; MetaQA-3 497.9 (Table 7) | WebQSP 94.9 %; MetaQA-3 99.0 % (Table 7) | no -- the graph entry is the linked entity; its RA variant unions LLM-retrieved paths, a different stage | answer Hits@1 / F1 after the LLM; the GNN's own retrieval recall is reported separately |
| NuTrea (NeurIPS 2023) | assigned topic entities, 2-hop subgraph (archaeology row; its statistics were not re-read here) | NOT STATED | NOT STATED | no | Hits@1 / F1 |
| GraphER | the base retriever's top-200 passages; the graph is candidate-induced at query time | 200 candidates | not reported in that form; its PR@K is coverage at K | yes -- the retrieval score is an input to the GAT and to GCS; the closest comparator to this phase | PR@5 / PR@10 |
| **ours (metaqa): every v2 arm and every frozen reference** | Dense + SPLADE seeds; base pool `equal_rrf_budget_50`; expansion `STRUCT:h3_c25_v2000` | 2,017.0 candidates | any-gold 0.9876, all-gold 0.8938, pool ceiling@5 0.8094 | yes — retrieval columns and the fixed base score | hit@1, R@5 |
| **ours (2wiki): every v2 arm and every frozen reference** | Dense + SPLADE seeds; base pool `equal_rrf_budget_50`; expansion `STRUCT:h1_c25` | 105.6 candidates | any-gold 0.9997, all-gold 0.9058, pool ceiling@5 0.9636 | yes — retrieval columns and the fixed base score | R@5 / R@10 |
| **ours (squad): every v2 arm and every frozen reference** | Dense + SPLADE seeds; base pool `equal_rrf_budget_50`; no graph expansion (retrieval-only pool) | 50.0 candidates | any-gold 0.9798, all-gold 0.9798, pool ceiling@5 0.9798 | yes — retrieval columns and the fixed base score | R@5 / R@10 |

## 10. Audit

| record | population digest = M3B | halves (gate / held) = filed | MRR audit | fixed rrf = M3B per query (max abs diff) | ceiling@5 as compiled |
|---|---|---|---|---|---|
| `2wiki` | True | 6,290 / 6,286: True | True (13 scorers) | 0.0e+00 | 0.9637 |
| `metaqa` | True | 19,738 / 19,400: True | True (13 scorers) | 0.0e+00 | 0.8094 |
| `squad` | True | 5,841 / 6,032: True | True (13 scorers) | 0.0e+00 | 0.9798 |

The CRAG package is read-only; the foreign `canonical.cpython-313.pyc` (37,614 B) is reported in every run's note and never removed. No population statistic was estimated and stored; the input block sees raw columns and within-query z-scores only.

## 11. Reading

Each hypothesis was filed before any fit (`hypotheses`) and is read on its declared measurement; `holds` is a reading of the filed claim, nothing is re-thresholded. Forbidden framings are not used: 'prove message passing is unnecessary', 'show that we do not need message passing', 'demonstrate that the MLP wins', 'describe the READ / NOT_READ band threshold of M3B as pre-registered (methodological_ruling_band_rule below)', 'architecture shopping -- fitting any candidate beyond the arms declared here without a dated amendment filed before its first fit'.

- **H_pilot** (V2_GATE): the selected GNN passes gnn_gate and the selected twin passes twin_gate — does not hold; outcome {'u_gnn_v2_ef': 'FAIL', 'u_mlp_v2_mix': 'FAIL'}; seed confirmation none (no pass).
- **H_operator_vs_basis**: on metaqa the selected GNN beats u_gnn_v2_core78 (the operator needs the basis) AND u_gnn_v2_core78 beats gat_universal_v1_trio (the operator alone beats the ordinary GAT), both on hit@1 with intervals excluding zero — V2_HELD_CONFIRMATION: holds (GNN − core78 hit@1 +0.013 [+0.009, +0.017]; core78 − trio +0.048 [+0.043, +0.052]); V2_GATE: holds (GNN − core78 hit@1 +0.011 [+0.007, +0.014]; core78 − trio +0.051 [+0.047, +0.056]).
- **H_evidence_flow**: u_gnn_v2_ef exceeds u_gnn_v2 on metaqa 3-hop hit@1 and on the multi-gold full_coverage@5 of 2wiki, and its evidence gate g2 is lower on squad than on metaqa at every step — V2_HELD_CONFIRMATION: does not hold (3hop hit@1 EF 0.8620 vs 0.8637; 2wiki multi-gold FC@5 0.8017 vs 0.8031; g2 per step squad 0.000, 0.000, 0.000; metaqa 0.713, 0.362, 0.618); V2_GATE: holds (3hop hit@1 EF 0.8679 vs 0.8633; 2wiki multi-gold FC@5 0.8018 vs 0.7997; g2 per step squad 0.000, 0.000, 0.000; metaqa 0.713, 0.362, 0.619).
- **H_do_nothing**: the passing arms hold the fixed rrf on squad within 0.005 and their delta_s magnitude on squad is the smallest of the trio — V2_HELD_CONFIRMATION: not readable (no passing arm); V2_GATE: not readable (no passing arm).

**Family outcome** (amendment 2 family_status_vocabulary): GNN_GATE FAIL, TWIN_GATE FAIL, overall **PILOT_FAILED** -- neither family passed its gate at seed 0; the proposed universal pair has not passed the pilot gate.

**Terminal state** (amendment 3 terminal_state_vocabulary): GNN_GATE GATE_FAIL, TWIN_GATE GATE_FAIL, terminal **PILOT_FAILED** -- no family is confirmed; the proposed universal pair is not confirmed and no family is rescued.

What this pilot is: whatever the pilot gate reads for the selected GNN and the selected twin on the three anchor datasets, and whatever the later stages measure if they open; a failed gate is a valid outcome and closes the phase with its numbers. What it is not: a result on test data (none was read), a statement about message passing (M3B answers the paper's question), a comparison to the published systems beyond the diagnostic calibration of section 9, or a six-dataset result (later_stages need their own dated authorization).

## 12. Run record

Rendered 2026-09-21T04:59:39Z by `scripts/universal_v2_report.py --stage doc` from the files below (sha256 of every sidecar a number above cites; the sidecars are gitignored, the record is committed in the declaration's run_record block). The held record is read, never recomputed.

| file | sha256 |
|---|---|
| `configs/universal_v2.yaml` | `d2ed6f5dd9873f5291b5aa79b8e31aad3315ae751b6b5f17ee44839cadf1ce16` |
| `configs/m3b_controlled_comparison.yaml` | `9c0e6e21e2fc8a88f35419e10e87f83b9018db0104500dad27a09de3dc10d09d` |
| `outputs/universal_v2/held_record.json` | `862acbadaf72dbe27b0f1f691259bb36802268dc728ba5d2f04a6b43ccc27adc` |
| `outputs/universal_v2/gate_record.json` | `8dcc217b08a75723af336e8d32ced4621aeed5f4e17cb1cfdb526433497d16e5` |
| `outputs/universal_v2/selection.json` | `3ed48ae3d04334ba367cb81b8b363d32db0e2bf746e4f692d7af0936bea4e3b6` |
| `outputs/universal_v2/timing.json` | `aaf7823452915354179bd79e98722f84b06f8b62a3a9a3f79960820b37c33ba1` |
| `outputs/universal_v2/feature_screen.json` | `c58b317e9ae0f34523e20b3b727e4394b16eaa1e9d836d34aa3730d08403551f` |
| `outputs/universal_v2/feature_contract.json` | `21d30260c7ef71c7cc217c2b32054625780db7e6b40c79d3c03fded658109e91` |
| `outputs/universal_v2/fits/gat_universal_v1_trio__H128__s0.json` | `0a946b1dda0b12eaa378210714b60eb9601c2d5f2367b922c67eb01c49f552ee` |
| `outputs/universal_v2/fits/u_gnn_v2__H128__s0.json` | `6378d64a734e68fda1ee9ea28eefe1d3368a35860b11480b160b0c00eecdf135` |
| `outputs/universal_v2/fits/u_gnn_v2_core78__H128__s0.json` | `02fb61575f4c300b47a41792cf475f8ec5c31eaa2d757563ffeff7f88342237a` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s0.json` | `bfd3e8e5957b259b2d8b38f85008f659f48e00eb79cdf43220aeb39817993009` |
| `outputs/universal_v2/fits/u_mlp_v2__H128__s0.json` | `74a50cc1657964321f31c85bdd889d0d732d75bc305aaf008d899a7c8ca812d7` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s0.json` | `21fab0efc9c7538c08de6311d3ebdde0944f49543e17bd0487f42e50f68940fb` |
| `outputs/universal_v2/eval/2wiki.json` | `ab7ad34de918fafb9fac7f46eebff014e14a741394e786b2929a3af92858890f` |
| `outputs/universal_v2/eval/2wiki.npz` | `6d8c1dcd2f815cf8c5e67664ace3ceaada3f54c6625ceeee768d35b04f13b4c2` |
| `outputs/universal_v2/eval/2wiki_query_ids.json` | `d439a33c22f39186f8e54f596aeb2bcb83a5d27745a0dc9285e8990f16923cd2` |
| `outputs/universal_v2/eval/metaqa.json` | `3db26c272e8d83c92c46392647da3c7ee0747f3561afceee32bb84b2d7824676` |
| `outputs/universal_v2/eval/metaqa.npz` | `380e0288b0e048cd18d83f06defd5f794fe7b7514a741bffbfbd6289e49eab5c` |
| `outputs/universal_v2/eval/metaqa_query_ids.json` | `b491d9e90e786c1d356d2a24142ec1b810e657efdfadff85a44ff1b489618692` |
| `outputs/universal_v2/eval/squad.json` | `8d7778ce4168b27fab4f187354bcc2336f0080ae6771977f386899b905641111` |
| `outputs/universal_v2/eval/squad.npz` | `cda64047bea803936ce811e5fa0d006ebffa10473e7baed44f2d203bd55121f3` |
| `outputs/universal_v2/eval/squad_query_ids.json` | `3ce32ec30d8920c8e0be25c2d53e325b29e4d16a5b16f7dfc9f564eb86d8a93c` |
| `outputs/m3b/eval/metaqa.npz` | `a4739cf66a8d0eb4a751abb0e9480305986eceb3876aca7662a05c7f0dfb97dd` |
| `outputs/m3b/eval/2wiki.npz` | `d2196c2f7769512071f1a9592504be6e1bdeff9729f1b2bcfae45f1ebdd76313` |
| `outputs/m3b/eval/squad.npz` | `c14fe820295dc18374ef8f0d5775f2d422af830bc9a579f527736929aeef8385` |

