# M3A-COMPILATION — headroom over the served canonical substrate

Registered question: *After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?*

This is the headroom half of the compilation report: **ceilings of candidate pools, not results of any ranker.** No model is scored here; nothing here is a scientific result about message passing. Every number is a property of a pool of candidates on the served substrate, measured on a labelled non-test split, and is reported with its K, its pool provenance, its dataset and its split (K semantics R1–R6, `configs/m3a_compilation.yaml#amendment_1_2026_09_13.k_semantics_adopted`). `pool_ceiling@K` is the cache-bounded ceiling; the corpus ceiling is a column and is 1.0 by construction only for the five text corpora.

## Provenance

| field | value |
|---|---|
| declared under | `configs/m3a_headroom.yaml` as run (sha256 `3b3109ccc1028014…`, the f2d15ac version; the file has since gained `run_record`, live sha256 `81c691b1ffb7c32d…`), `configs/m3a_compilation.yaml#amendment_1_2026_09_13.headroom_amended` |
| git commit of the rule | `f2d15ac` (2026-09-12T23:53:48Z), before the first run |
| runner | `scripts/m3a_headroom.py` (sha256 `d3df956c55f852fb…`) |
| served freeze RECORD_SHA256 | `58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0` |
| git commit at run time | `f2d15ac9b335221d2e31d77137d6f49922527da2-dirty` |
| run UTC | 2026-09-13T00:02:31Z, 2026-09-13T00:06:28Z, 2026-09-13T00:14:02Z, 2026-09-13T00:28:36Z, 2026-09-13T00:36:10Z, 2026-09-13T00:55:40Z |
| package access | READ_ONLY; outputs are a sidecar under `outputs/m3a/headroom/` (gitignored; sha256s filed in the config) |
| GPU | none |

## Populations

One labelled non-test split per dataset, taken whole. WebQSP is scored on the seed-free `train_holdout` carve (lexicographic stride 2 of train, 1,549 ids, sha pinned in the config); its test split is barred. The corpus ceiling is a **column**: on webqsp it is below one because the gold references of RoG resolve at the reference level to only part of the served corpus, so no webqsp ceiling may be averaged with a text-corpus ceiling. LEGACY_CONTINUITY rows are the only bridge to the M0A–M2D numbers and are **not an evaluation population** (train-derived on four of five datasets).

| dataset | eval split | queries in split | zero-gold excluded | queries scored | corpus ceiling, reference level | corpus ceiling, query level (any gold) | note | LEGACY_CONTINUITY rows (NOT a population) |
|---|---|---|---|---|---|---|---|---|
| metaqa | dev | 39,138 | 0 | 39,138 | 1.0000 | 1.0000 | text corpus: gold resolves by construction | 1,998 (dev 1,998) |
| squad | dev | 11,873 | 0 | 11,873 | 1.0000 | 1.0000 | text corpus: gold resolves by construction | 2,000 (train 2,000) |
| musique | dev | 2,417 | 0 | 2,417 | 1.0000 | 1.0000 | text corpus: gold resolves by construction | 2,000 (train 2,000) |
| hotpotqa | validation | 7,405 | 0 | 7,405 | 1.0000 | 1.0000 | text corpus: gold resolves by construction | 2,000 (train 1,842, validation 158) |
| 2wiki | dev | 12,576 | 0 | 12,576 | 1.0000 | 1.0000 | text corpus: gold resolves by construction | 2,000 (train 2,000) |
| webqsp | train_holdout | 1,549 | 46 | 1,503 | 0.5255 | 0.9703 | KB corpus: 11,760 of 22,379 gold references resolve; of answerable 0.9792; classes {"FULL": 1462, "PARTIAL": 41, "NONE_IN_CORPUS": 32, "NO_GOLD_GIVEN": 14} | – |

## Retrieval depth curve (R4)

For a prefix pool, `pool_ceiling@K` at cut-off K equals the recall@K of the retriever itself, so this table is both the recall curve of each served cache and the ceiling of the pool it would define. Equal-RRF fuses the two top-200 lists and therefore stops at 400.


### metaqa — `dev`, 39,138 queries scored

| K (= prefix depth) | dense recall@K | splade recall@K | equal-RRF recall@K |
|---|---|---|---|
| 1 | 0.0007 | 0.0002 | 0.0003 |
| 5 | 0.0085 | 0.0025 | 0.0056 |
| 10 | 0.0132 | 0.0039 | 0.0101 |
| 20 | 0.0199 | 0.0057 | 0.0157 |
| 50 | 0.0320 | 0.0095 | 0.0268 |
| 100 | 0.0473 | 0.0219 | 0.0393 |
| 200 | 0.0682 | 0.0664 | 0.0661 |
| 400 | 0.0974 | 0.1583 | 0.1254 |
| 1000 | 0.1540 | 0.2926 | – |

### squad — `dev`, 11,873 queries scored

| K (= prefix depth) | dense recall@K | splade recall@K | equal-RRF recall@K |
|---|---|---|---|
| 1 | 0.6939 | 0.7270 | 0.7349 |
| 5 | 0.8769 | 0.8889 | 0.9053 |
| 10 | 0.9182 | 0.9229 | 0.9398 |
| 20 | 0.9470 | 0.9450 | 0.9629 |
| 50 | 0.9694 | 0.9673 | 0.9798 |
| 100 | 0.9805 | 0.9784 | 0.9880 |
| 200 | 0.9880 | 0.9848 | 0.9933 |
| 400 | 0.9931 | 0.9907 | 0.9955 |
| 1000 | 0.9976 | 0.9950 | – |

### musique — `dev`, 2,417 queries scored

| K (= prefix depth) | dense recall@K | splade recall@K | equal-RRF recall@K |
|---|---|---|---|
| 1 | 0.2496 | 0.2575 | 0.2606 |
| 5 | 0.4572 | 0.4314 | 0.4733 |
| 10 | 0.5299 | 0.4958 | 0.5490 |
| 20 | 0.5991 | 0.5527 | 0.6167 |
| 50 | 0.6741 | 0.6191 | 0.6888 |
| 100 | 0.7285 | 0.6621 | 0.7370 |
| 200 | 0.7788 | 0.7096 | 0.7809 |
| 400 | 0.8227 | 0.7534 | 0.8131 |
| 1000 | 0.8725 | 0.8088 | – |

### hotpotqa — `validation`, 7,405 queries scored

| K (= prefix depth) | dense recall@K | splade recall@K | equal-RRF recall@K |
|---|---|---|---|
| 1 | 0.4008 | 0.4005 | 0.4145 |
| 5 | 0.6577 | 0.6499 | 0.6843 |
| 10 | 0.7103 | 0.6971 | 0.7375 |
| 20 | 0.7507 | 0.7405 | 0.7837 |
| 50 | 0.7965 | 0.7858 | 0.8303 |
| 100 | 0.8257 | 0.8151 | 0.8569 |
| 200 | 0.8531 | 0.8412 | 0.8770 |
| 400 | 0.8795 | 0.8648 | 0.8940 |
| 1000 | 0.9103 | 0.8945 | – |

### 2wiki — `dev`, 12,576 queries scored

| K (= prefix depth) | dense recall@K | splade recall@K | equal-RRF recall@K |
|---|---|---|---|
| 1 | 0.3269 | 0.3860 | 0.3880 |
| 5 | 0.5171 | 0.6055 | 0.6066 |
| 10 | 0.5714 | 0.6457 | 0.6558 |
| 20 | 0.6149 | 0.6778 | 0.6926 |
| 50 | 0.6626 | 0.7065 | 0.7254 |
| 100 | 0.6925 | 0.7252 | 0.7429 |
| 200 | 0.7195 | 0.7421 | 0.7563 |
| 400 | 0.7453 | 0.7581 | 0.7698 |
| 1000 | 0.7759 | 0.7807 | – |

### webqsp — `train_holdout`, 1,503 queries scored

| K (= prefix depth) | dense recall@K | splade recall@K | equal-RRF recall@K |
|---|---|---|---|
| 1 | 0.0456 | 0.0191 | 0.0242 |
| 5 | 0.1095 | 0.0367 | 0.0535 |
| 10 | 0.1377 | 0.0466 | 0.0705 |
| 20 | 0.1775 | 0.0595 | 0.1014 |
| 50 | 0.2315 | 0.0859 | 0.1725 |
| 100 | 0.2946 | 0.1089 | 0.2478 |
| 200 | 0.3683 | 0.1403 | 0.3239 |
| 400 | 0.4359 | 0.1677 | 0.3772 |
| 1000 | 0.5085 | 0.2080 | – |

## Retrieval pools (the inherited seven)

`dense_top200`, `splade_top200`, their union (`frozen_union`) and the equal-RRF budget pools, exactly as `configs/candidate_headroom.yaml` defined them. `fraction of attainable@5` divides `pool_ceiling@5` by the ceiling a perfect retriever would have at K=5 given the gold count (`recall_ceiling_perfect_retrieval@5`).


### metaqa

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.0553 | 0.0669 | 0.0682 | 0.0809 | 0.1853 | 0.0352 | 0.0352 |
| splade_top200 | 200.0 (200) | 0.0576 | 0.0663 | 0.0664 | 0.0801 | 0.1202 | 0.0469 | 0.0469 |
| frozen_union | 377.8 (396) | 0.1038 | 0.1240 | 0.1254 | 0.1498 | 0.2772 | 0.0761 | 0.0761 |
| equal_rrf_budget_50 | 50.0 (50) | 0.0244 | 0.0267 | 0.0268 | 0.0323 | 0.0787 | 0.0154 | 0.0154 |
| equal_rrf_budget_100 | 100.0 (100) | 0.0349 | 0.0392 | 0.0393 | 0.0473 | 0.1143 | 0.0223 | 0.0223 |
| equal_rrf_budget_200 | 200.0 (200) | 0.0579 | 0.0656 | 0.0661 | 0.0793 | 0.1717 | 0.0378 | 0.0378 |
| equal_rrf_budget_400 | 377.8 (396) | 0.1038 | 0.1240 | 0.1254 | 0.1498 | 0.2772 | 0.0761 | 0.0761 |

### squad

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.9880 | 0.9880 | 0.9880 | 0.9880 | 0.9880 | 0.9880 | 0.9880 |
| splade_top200 | 200.0 (200) | 0.9848 | 0.9848 | 0.9848 | 0.9848 | 0.9848 | 0.9848 | 0.9848 |
| frozen_union | 321.8 (367) | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 |
| equal_rrf_budget_50 | 50.0 (50) | 0.9798 | 0.9798 | 0.9798 | 0.9798 | 0.9798 | 0.9798 | 0.9798 |
| equal_rrf_budget_100 | 100.0 (100) | 0.9880 | 0.9880 | 0.9880 | 0.9880 | 0.9880 | 0.9880 | 0.9880 |
| equal_rrf_budget_200 | 200.0 (200) | 0.9933 | 0.9933 | 0.9933 | 0.9933 | 0.9933 | 0.9933 | 0.9933 |
| equal_rrf_budget_400 | 321.8 (367) | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 |

### musique

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.3996 | 0.7788 | 0.7788 | 0.7788 | 0.9855 | 0.5391 | 0.5391 |
| splade_top200 | 200.0 (200) | 0.3991 | 0.7096 | 0.7096 | 0.7096 | 0.9847 | 0.3922 | 0.3922 |
| frozen_union | 340.7 (380) | 0.4027 | 0.8131 | 0.8131 | 0.8131 | 0.9930 | 0.5850 | 0.5850 |
| equal_rrf_budget_50 | 50.0 (50) | 0.3960 | 0.6888 | 0.6888 | 0.6888 | 0.9768 | 0.3765 | 0.3765 |
| equal_rrf_budget_100 | 100.0 (100) | 0.3999 | 0.7370 | 0.7370 | 0.7370 | 0.9863 | 0.4497 | 0.4497 |
| equal_rrf_budget_200 | 200.0 (200) | 0.4016 | 0.7809 | 0.7809 | 0.7809 | 0.9905 | 0.5263 | 0.5263 |
| equal_rrf_budget_400 | 340.7 (380) | 0.4027 | 0.8131 | 0.8131 | 0.8131 | 0.9930 | 0.5850 | 0.5850 |

### hotpotqa

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.4941 | 0.8531 | 0.8531 | 0.8531 | 0.9883 | 0.7179 | 0.7179 |
| splade_top200 | 200.0 (200) | 0.4959 | 0.8412 | 0.8412 | 0.8412 | 0.9918 | 0.6906 | 0.6906 |
| frozen_union | 354.4 (388) | 0.4982 | 0.8940 | 0.8940 | 0.8940 | 0.9965 | 0.7915 | 0.7915 |
| equal_rrf_budget_50 | 50.0 (50) | 0.4944 | 0.8303 | 0.8303 | 0.8303 | 0.9888 | 0.6718 | 0.6718 |
| equal_rrf_budget_100 | 100.0 (100) | 0.4960 | 0.8569 | 0.8569 | 0.8569 | 0.9920 | 0.7217 | 0.7217 |
| equal_rrf_budget_200 | 200.0 (200) | 0.4974 | 0.8770 | 0.8770 | 0.8770 | 0.9947 | 0.7594 | 0.7594 |
| equal_rrf_budget_400 | 354.4 (388) | 0.4982 | 0.8940 | 0.8940 | 0.8940 | 0.9965 | 0.7915 | 0.7915 |

### 2wiki

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.4437 | 0.7195 | 0.7195 | 0.7195 | 0.9967 | 0.4382 | 0.4382 |
| splade_top200 | 200.0 (200) | 0.4450 | 0.7421 | 0.7421 | 0.7421 | 0.9994 | 0.4636 | 0.4636 |
| frozen_union | 364.5 (392) | 0.4453 | 0.7698 | 0.7698 | 0.7698 | 1.0000 | 0.5128 | 0.5128 |
| equal_rrf_budget_50 | 50.0 (50) | 0.4451 | 0.7254 | 0.7254 | 0.7254 | 0.9996 | 0.4335 | 0.4335 |
| equal_rrf_budget_100 | 100.0 (100) | 0.4452 | 0.7429 | 0.7429 | 0.7429 | 0.9998 | 0.4650 | 0.4650 |
| equal_rrf_budget_200 | 200.0 (200) | 0.4453 | 0.7563 | 0.7563 | 0.7563 | 1.0000 | 0.4885 | 0.4885 |
| equal_rrf_budget_400 | 364.5 (392) | 0.4453 | 0.7698 | 0.7698 | 0.7698 | 1.0000 | 0.5128 | 0.5128 |

### webqsp

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.3016 | 0.3576 | 0.3674 | 0.4033 | 0.5063 | 0.2788 | 0.2788 |
| splade_top200 | 200.0 (200) | 0.1158 | 0.1367 | 0.1398 | 0.1542 | 0.2169 | 0.1011 | 0.1011 |
| frozen_union | 327.2 (377) | 0.3047 | 0.3655 | 0.3761 | 0.4123 | 0.5136 | 0.2854 | 0.2854 |
| equal_rrf_budget_50 | 50.0 (50) | 0.1494 | 0.1702 | 0.1722 | 0.1919 | 0.2628 | 0.1311 | 0.1311 |
| equal_rrf_budget_100 | 100.0 (100) | 0.2116 | 0.2434 | 0.2471 | 0.2745 | 0.3593 | 0.1876 | 0.1876 |
| equal_rrf_budget_200 | 200.0 (200) | 0.2673 | 0.3156 | 0.3231 | 0.3561 | 0.4544 | 0.2415 | 0.2415 |
| equal_rrf_budget_400 | 327.2 (377) | 0.3047 | 0.3655 | 0.3761 | 0.4123 | 0.5136 | 0.2854 | 0.2854 |

## Graph regimes: what a fixed, parameter-free expansion adds to a retrieval pool

Seeds are the dense top-5 then the splade top-5 of the served caches (never gold, never assigned topic entities). `h1_c25`/`h1_c100`: one undirected hop, the first 25/100 neighbours of each seed per family (weighted families by weight descending, structural by position). `h2_c25`: a second hop from the hop-1 nodes in order of first appearance, 25 per frontier node per family, stopping at 2,000 visited nodes. The expansion is a function of seeds and graph alone and is unioned with the base pool. `FULL` is the union of the three per-family expansions, so its exposure is up to three times that of a single family. `missing golds recovered` and `recovered per added candidate` are `pool_movement` against the base pool.


### metaqa

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 103.6 (137) | 0.3771 | 0.3823 | 0.4554 | 0.5703 | 0.3185 | 36605 | 0.0174 |
| STRUCT | h1_c25 | rrf@200 | 252.9 (286) | 0.3975 | 0.4032 | 0.4801 | 0.6178 | 0.3291 | 35046 | 0.0169 |
| STRUCT | h1_c100 | rrf@50 | 105.2 (143) | 0.3775 | 0.3828 | 0.4559 | 0.5709 | 0.3192 | 37075 | 0.0172 |
| STRUCT | h1_c100 | rrf@200 | 254.4 (292) | 0.3979 | 0.4036 | 0.4805 | 0.6181 | 0.3298 | 35504 | 0.0167 |
| STRUCT | h2_c25 | rrf@50 | 618.7 (959) | 0.7102 | 0.7632 | 0.8577 | 0.9109 | 0.6740 | 137696 | 0.0062 |
| STRUCT | h2_c25 | rrf@200 | 762.0 (1098) | 0.7136 | 0.7671 | 0.8619 | 0.9223 | 0.6749 | 133340 | 0.0061 |
| NER | h1_c25 | rrf@50 | 51.7 (56) | 0.0296 | 0.0297 | 0.0358 | 0.0837 | 0.0176 | 233 | 0.0036 |
| NER | h1_c25 | rrf@200 | 201.4 (205) | 0.0674 | 0.0678 | 0.0814 | 0.1744 | 0.0391 | 167 | 0.0031 |
| NER | h1_c100 | rrf@50 | 51.7 (56) | 0.0296 | 0.0297 | 0.0358 | 0.0837 | 0.0176 | 233 | 0.0036 |
| NER | h1_c100 | rrf@200 | 201.4 (205) | 0.0674 | 0.0678 | 0.0814 | 0.1744 | 0.0391 | 167 | 0.0031 |
| NER | h2_c25 | rrf@50 | 52.1 (58) | 0.0296 | 0.0297 | 0.0358 | 0.0838 | 0.0176 | 238 | 0.0030 |
| NER | h2_c25 | rrf@200 | 201.8 (207) | 0.0674 | 0.0679 | 0.0814 | 0.1745 | 0.0391 | 172 | 0.0025 |
| KNN | h1_c25 | rrf@50 | 75.9 (95) | 0.0358 | 0.0359 | 0.0432 | 0.0996 | 0.0210 | 1251 | 0.0012 |
| KNN | h1_c25 | rrf@200 | 219.0 (237) | 0.0703 | 0.0708 | 0.0848 | 0.1812 | 0.0408 | 718 | 0.0010 |
| KNN | h1_c100 | rrf@50 | 76.2 (98) | 0.0358 | 0.0359 | 0.0432 | 0.0997 | 0.0210 | 1253 | 0.0012 |
| KNN | h1_c100 | rrf@200 | 219.2 (240) | 0.0703 | 0.0708 | 0.0849 | 0.1813 | 0.0408 | 720 | 0.0010 |
| KNN | h2_c25 | rrf@50 | 182.0 (275) | 0.0536 | 0.0538 | 0.0648 | 0.1441 | 0.0302 | 3979 | 0.0008 |
| KNN | h2_c25 | rrf@200 | 314.4 (403) | 0.0798 | 0.0805 | 0.0964 | 0.2045 | 0.0455 | 2534 | 0.0006 |
| FULL | h1_c25 | rrf@50 | 130.1 (171) | 0.3800 | 0.3853 | 0.4590 | 0.5784 | 0.3199 | 37423 | 0.0119 |
| FULL | h1_c25 | rrf@200 | 272.4 (311) | 0.3991 | 0.4049 | 0.4820 | 0.6216 | 0.3299 | 35555 | 0.0126 |
| FULL | h1_c100 | rrf@50 | 131.9 (177) | 0.3804 | 0.3857 | 0.4595 | 0.5788 | 0.3206 | 37889 | 0.0118 |
| FULL | h1_c100 | rrf@200 | 274.2 (318) | 0.3994 | 0.4053 | 0.4824 | 0.6219 | 0.3306 | 36011 | 0.0124 |
| FULL | h2_c25 | rrf@50 | 1166.9 (1718) | 0.7444 | 0.8103 | 0.8991 | 0.9448 | 0.7033 | 157280 | 0.0036 |
| FULL | h2_c25 | rrf@200 | 1290.4 (1834) | 0.7469 | 0.8134 | 0.9021 | 0.9505 | 0.7039 | 152436 | 0.0036 |

### squad

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 131.4 (199) | 0.9804 | 0.9804 | 0.9804 | 0.9804 | 0.9804 | 7 | 0.0000 |
| STRUCT | h1_c25 | rrf@200 | 275.8 (345) | 0.9933 | 0.9933 | 0.9933 | 0.9933 | 0.9933 | 1 | 0.0000 |
| STRUCT | h1_c100 | rrf@50 | 265.9 (482) | 0.9815 | 0.9815 | 0.9815 | 0.9815 | 0.9815 | 20 | 0.0000 |
| STRUCT | h1_c100 | rrf@200 | 401.8 (616) | 0.9935 | 0.9935 | 0.9935 | 0.9935 | 0.9935 | 3 | 0.0000 |
| STRUCT | h2_c25 | rrf@50 | 396.7 (726) | 0.9812 | 0.9812 | 0.9812 | 0.9812 | 0.9812 | 17 | 0.0000 |
| STRUCT | h2_c25 | rrf@200 | 535.1 (857) | 0.9936 | 0.9936 | 0.9936 | 0.9936 | 0.9936 | 4 | 0.0000 |
| NER | h1_c25 | rrf@50 | 113.0 (185) | 0.9821 | 0.9821 | 0.9821 | 0.9821 | 0.9821 | 28 | 0.0000 |
| NER | h1_c25 | rrf@200 | 254.1 (321) | 0.9935 | 0.9935 | 0.9935 | 0.9935 | 0.9935 | 3 | 0.0000 |
| NER | h1_c100 | rrf@50 | 134.7 (254) | 0.9826 | 0.9826 | 0.9826 | 0.9826 | 0.9826 | 34 | 0.0000 |
| NER | h1_c100 | rrf@200 | 273.9 (386) | 0.9936 | 0.9936 | 0.9936 | 0.9936 | 0.9936 | 4 | 0.0000 |
| NER | h2_c25 | rrf@50 | 739.0 (1541) | 0.9859 | 0.9859 | 0.9859 | 0.9859 | 0.9859 | 73 | 0.0000 |
| NER | h2_c25 | rrf@200 | 860.3 (1638) | 0.9943 | 0.9943 | 0.9943 | 0.9943 | 0.9943 | 12 | 0.0000 |
| KNN | h1_c25 | rrf@50 | 63.9 (81) | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 14 | 0.0001 |
| KNN | h1_c25 | rrf@200 | 207.8 (223) | 0.9934 | 0.9934 | 0.9934 | 0.9934 | 0.9934 | 2 | 0.0000 |
| KNN | h1_c100 | rrf@50 | 63.9 (81) | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 14 | 0.0001 |
| KNN | h1_c100 | rrf@200 | 207.8 (223) | 0.9934 | 0.9934 | 0.9934 | 0.9934 | 0.9934 | 2 | 0.0000 |
| KNN | h2_c25 | rrf@50 | 107.0 (167) | 0.9832 | 0.9832 | 0.9832 | 0.9832 | 0.9832 | 40 | 0.0001 |
| KNN | h2_c25 | rrf@200 | 240.4 (297) | 0.9941 | 0.9941 | 0.9941 | 0.9941 | 0.9941 | 10 | 0.0000 |
| FULL | h1_c25 | rrf@50 | 200.7 (322) | 0.9834 | 0.9834 | 0.9834 | 0.9834 | 0.9834 | 43 | 0.0000 |
| FULL | h1_c25 | rrf@200 | 333.2 (451) | 0.9937 | 0.9937 | 0.9937 | 0.9937 | 0.9937 | 5 | 0.0000 |
| FULL | h1_c100 | rrf@50 | 349.7 (617) | 0.9847 | 0.9847 | 0.9847 | 0.9847 | 0.9847 | 58 | 0.0000 |
| FULL | h1_c100 | rrf@200 | 473.8 (738) | 0.9939 | 0.9939 | 0.9939 | 0.9939 | 0.9939 | 8 | 0.0000 |
| FULL | h2_c25 | rrf@50 | 1703.0 (2029) | 0.9885 | 0.9885 | 0.9885 | 0.9885 | 0.9885 | 104 | 0.0000 |
| FULL | h2_c25 | rrf@200 | 1798.5 (2145) | 0.9954 | 0.9954 | 0.9954 | 0.9954 | 0.9954 | 25 | 0.0000 |

### musique

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 138.1 (199) | 0.7618 | 0.7618 | 0.7618 | 0.9801 | 0.5052 | 462 | 0.0022 |
| STRUCT | h1_c25 | rrf@200 | 283.2 (344) | 0.8302 | 0.8302 | 0.8302 | 0.9913 | 0.6189 | 307 | 0.0015 |
| STRUCT | h1_c100 | rrf@50 | 246.5 (458) | 0.7860 | 0.7860 | 0.7860 | 0.9835 | 0.5486 | 629 | 0.0013 |
| STRUCT | h1_c100 | rrf@200 | 386.1 (591) | 0.8430 | 0.8430 | 0.8430 | 0.9926 | 0.6487 | 393 | 0.0009 |
| STRUCT | h2_c25 | rrf@50 | 671.5 (1113) | 0.7965 | 0.7965 | 0.7965 | 0.9847 | 0.5594 | 701 | 0.0005 |
| STRUCT | h2_c25 | rrf@200 | 808.7 (1241) | 0.8495 | 0.8495 | 0.8495 | 0.9942 | 0.6545 | 437 | 0.0003 |
| NER | h1_c25 | rrf@50 | 137.2 (196) | 0.7952 | 0.7952 | 0.7952 | 0.9822 | 0.5647 | 651 | 0.0031 |
| NER | h1_c25 | rrf@200 | 279.2 (337) | 0.8593 | 0.8593 | 0.8593 | 0.9926 | 0.6711 | 473 | 0.0025 |
| NER | h1_c100 | rrf@50 | 167.6 (278) | 0.8055 | 0.8055 | 0.8055 | 0.9839 | 0.5776 | 725 | 0.0026 |
| NER | h1_c100 | rrf@200 | 307.8 (414) | 0.8664 | 0.8664 | 0.8664 | 0.9938 | 0.6864 | 528 | 0.0020 |
| NER | h2_c25 | rrf@50 | 986.3 (1678) | 0.8394 | 0.8394 | 0.8394 | 0.9863 | 0.6450 | 975 | 0.0004 |
| NER | h2_c25 | rrf@200 | 1111.9 (1805) | 0.8844 | 0.8844 | 0.8844 | 0.9942 | 0.7278 | 665 | 0.0003 |
| KNN | h1_c25 | rrf@50 | 74.8 (93) | 0.7409 | 0.7409 | 0.7409 | 0.9818 | 0.4642 | 338 | 0.0056 |
| KNN | h1_c25 | rrf@200 | 217.6 (237) | 0.8108 | 0.8108 | 0.8108 | 0.9926 | 0.5825 | 194 | 0.0046 |
| KNN | h1_c100 | rrf@50 | 74.8 (93) | 0.7409 | 0.7409 | 0.7409 | 0.9818 | 0.4642 | 338 | 0.0056 |
| KNN | h1_c100 | rrf@200 | 217.7 (237) | 0.8108 | 0.8108 | 0.8108 | 0.9926 | 0.5825 | 194 | 0.0045 |
| KNN | h2_c25 | rrf@50 | 166.2 (255) | 0.7725 | 0.7725 | 0.7725 | 0.9843 | 0.5230 | 554 | 0.0020 |
| KNN | h2_c25 | rrf@200 | 297.5 (389) | 0.8281 | 0.8281 | 0.8281 | 0.9934 | 0.6165 | 308 | 0.0013 |
| FULL | h1_c25 | rrf@50 | 238.7 (347) | 0.8377 | 0.8377 | 0.8377 | 0.9863 | 0.6446 | 949 | 0.0021 |
| FULL | h1_c25 | rrf@200 | 372.3 (477) | 0.8836 | 0.8836 | 0.8836 | 0.9934 | 0.7232 | 644 | 0.0015 |
| FULL | h1_c100 | rrf@50 | 369.9 (638) | 0.8563 | 0.8563 | 0.8563 | 0.9880 | 0.6781 | 1083 | 0.0014 |
| FULL | h1_c100 | rrf@200 | 498.3 (751) | 0.8950 | 0.8950 | 0.8950 | 0.9942 | 0.7493 | 729 | 0.0010 |
| FULL | h2_c25 | rrf@50 | 1985.4 (2036) | 0.8937 | 0.8937 | 0.8937 | 0.9905 | 0.7513 | 1355 | 0.0003 |
| FULL | h2_c25 | rrf@200 | 2092.4 (2172) | 0.9207 | 0.9207 | 0.9207 | 0.9950 | 0.8026 | 920 | 0.0002 |

### hotpotqa

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 93.7 (137) | 0.9800 | 0.9800 | 0.9800 | 0.9916 | 0.9684 | 2217 | 0.0069 |
| STRUCT | h1_c25 | rrf@200 | 239.3 (281) | 0.9866 | 0.9866 | 0.9866 | 0.9957 | 0.9774 | 1622 | 0.0056 |
| STRUCT | h1_c100 | rrf@50 | 108.9 (202) | 0.9828 | 0.9828 | 0.9828 | 0.9918 | 0.9738 | 2258 | 0.0052 |
| STRUCT | h1_c100 | rrf@200 | 252.3 (337) | 0.9881 | 0.9881 | 0.9881 | 0.9957 | 0.9806 | 1645 | 0.0042 |
| STRUCT | h2_c25 | rrf@50 | 634.1 (1138) | 0.9841 | 0.9841 | 0.9841 | 0.9930 | 0.9753 | 2278 | 0.0005 |
| STRUCT | h2_c25 | rrf@200 | 771.4 (1271) | 0.9890 | 0.9890 | 0.9890 | 0.9961 | 0.9819 | 1658 | 0.0004 |
| NER | h1_c25 | rrf@50 | 123.7 (194) | 0.8652 | 0.8652 | 0.8652 | 0.9903 | 0.7402 | 517 | 0.0009 |
| NER | h1_c25 | rrf@200 | 269.6 (338) | 0.9018 | 0.9018 | 0.9018 | 0.9951 | 0.8084 | 366 | 0.0007 |
| NER | h1_c100 | rrf@50 | 147.7 (277) | 0.8679 | 0.8679 | 0.8679 | 0.9903 | 0.7456 | 557 | 0.0008 |
| NER | h1_c100 | rrf@200 | 292.9 (421) | 0.9036 | 0.9036 | 0.9036 | 0.9951 | 0.8122 | 394 | 0.0006 |
| NER | h2_c25 | rrf@50 | 772.6 (1709) | 0.8777 | 0.8777 | 0.8777 | 0.9914 | 0.7641 | 702 | 0.0001 |
| NER | h2_c25 | rrf@200 | 913.2 (1850) | 0.9098 | 0.9098 | 0.9098 | 0.9953 | 0.8243 | 485 | 0.0001 |
| KNN | h1_c25 | rrf@50 | 78.0 (100) | 0.8554 | 0.8554 | 0.8554 | 0.9908 | 0.7199 | 371 | 0.0018 |
| KNN | h1_c25 | rrf@200 | 221.9 (243) | 0.8922 | 0.8922 | 0.8922 | 0.9958 | 0.7885 | 224 | 0.0014 |
| KNN | h1_c100 | rrf@50 | 78.1 (100) | 0.8554 | 0.8554 | 0.8554 | 0.9908 | 0.7199 | 371 | 0.0018 |
| KNN | h1_c100 | rrf@200 | 222.0 (243) | 0.8922 | 0.8922 | 0.8922 | 0.9958 | 0.7885 | 224 | 0.0014 |
| KNN | h2_c25 | rrf@50 | 195.6 (306) | 0.8710 | 0.8710 | 0.8710 | 0.9918 | 0.7502 | 602 | 0.0006 |
| KNN | h2_c25 | rrf@200 | 330.7 (441) | 0.9027 | 0.9027 | 0.9027 | 0.9961 | 0.8093 | 380 | 0.0004 |
| FULL | h1_c25 | rrf@50 | 179.0 (276) | 0.9822 | 0.9822 | 0.9822 | 0.9923 | 0.9720 | 2249 | 0.0024 |
| FULL | h1_c25 | rrf@200 | 317.7 (414) | 0.9878 | 0.9878 | 0.9878 | 0.9961 | 0.9796 | 1641 | 0.0019 |
| FULL | h1_c100 | rrf@50 | 215.3 (386) | 0.9843 | 0.9843 | 0.9843 | 0.9923 | 0.9764 | 2281 | 0.0019 |
| FULL | h1_c100 | rrf@200 | 351.9 (519) | 0.9891 | 0.9891 | 0.9891 | 0.9961 | 0.9822 | 1660 | 0.0015 |
| FULL | h2_c25 | rrf@50 | 1798.8 (2037) | 0.9886 | 0.9886 | 0.9886 | 0.9950 | 0.9822 | 2344 | 0.0002 |
| FULL | h2_c25 | rrf@200 | 1919.0 (2178) | 0.9918 | 0.9918 | 0.9918 | 0.9970 | 0.9865 | 1699 | 0.0001 |

### 2wiki

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 105.6 (144) | 0.9636 | 0.9636 | 0.9636 | 0.9997 | 0.9058 | 8124 | 0.0116 |
| STRUCT | h1_c25 | rrf@200 | 253.2 (291) | 0.9701 | 0.9701 | 0.9701 | 1.0000 | 0.9177 | 7441 | 0.0111 |
| STRUCT | h1_c100 | rrf@50 | 112.9 (182) | 0.9646 | 0.9646 | 0.9646 | 0.9997 | 0.9080 | 8154 | 0.0103 |
| STRUCT | h1_c100 | rrf@200 | 260.0 (325) | 0.9709 | 0.9709 | 0.9709 | 1.0000 | 0.9194 | 7463 | 0.0099 |
| STRUCT | h2_c25 | rrf@50 | 1006.9 (1637) | 0.9701 | 0.9701 | 0.9701 | 0.9998 | 0.9198 | 8310 | 0.0007 |
| STRUCT | h2_c25 | rrf@200 | 1149.6 (1779) | 0.9755 | 0.9755 | 0.9755 | 1.0000 | 0.9295 | 7596 | 0.0006 |
| NER | h1_c25 | rrf@50 | 156.4 (219) | 0.8249 | 0.8249 | 0.8249 | 0.9997 | 0.6023 | 3183 | 0.0024 |
| NER | h1_c25 | rrf@200 | 303.1 (366) | 0.8418 | 0.8418 | 0.8418 | 1.0000 | 0.6340 | 2799 | 0.0022 |
| NER | h1_c100 | rrf@50 | 213.7 (349) | 0.8365 | 0.8365 | 0.8365 | 0.9997 | 0.6231 | 3577 | 0.0017 |
| NER | h1_c100 | rrf@200 | 359.7 (495) | 0.8524 | 0.8524 | 0.8524 | 1.0000 | 0.6528 | 3163 | 0.0016 |
| NER | h2_c25 | rrf@50 | 1430.3 (2040) | 0.8558 | 0.8558 | 0.8558 | 0.9998 | 0.6615 | 4118 | 0.0002 |
| NER | h2_c25 | rrf@200 | 1571.5 (2187) | 0.8682 | 0.8682 | 0.8682 | 1.0000 | 0.6845 | 3611 | 0.0002 |
| KNN | h1_c25 | rrf@50 | 79.6 (99) | 0.7657 | 0.7657 | 0.7657 | 0.9996 | 0.5049 | 1163 | 0.0031 |
| KNN | h1_c25 | rrf@200 | 225.3 (244) | 0.7854 | 0.7854 | 0.7854 | 1.0000 | 0.5396 | 857 | 0.0027 |
| KNN | h1_c100 | rrf@50 | 79.6 (99) | 0.7657 | 0.7657 | 0.7657 | 0.9996 | 0.5049 | 1163 | 0.0031 |
| KNN | h1_c100 | rrf@200 | 225.3 (244) | 0.7854 | 0.7854 | 0.7854 | 1.0000 | 0.5396 | 857 | 0.0027 |
| KNN | h2_c25 | rrf@50 | 202.1 (300) | 0.7912 | 0.7912 | 0.7912 | 0.9997 | 0.5506 | 1890 | 0.0010 |
| KNN | h2_c25 | rrf@200 | 342.5 (440) | 0.8061 | 0.8061 | 0.8061 | 1.0000 | 0.5763 | 1456 | 0.0008 |
| FULL | h1_c25 | rrf@50 | 221.6 (308) | 0.9726 | 0.9726 | 0.9726 | 0.9998 | 0.9257 | 8385 | 0.0039 |
| FULL | h1_c25 | rrf@200 | 364.2 (451) | 0.9772 | 0.9772 | 0.9772 | 1.0000 | 0.9338 | 7651 | 0.0037 |
| FULL | h1_c100 | rrf@50 | 282.3 (446) | 0.9739 | 0.9739 | 0.9739 | 0.9998 | 0.9284 | 8425 | 0.0029 |
| FULL | h1_c100 | rrf@200 | 424.0 (587) | 0.9782 | 0.9782 | 0.9782 | 1.0000 | 0.9361 | 7684 | 0.0027 |
| FULL | h2_c25 | rrf@50 | 2007.4 (2040) | 0.9807 | 0.9807 | 0.9807 | 0.9998 | 0.9441 | 8633 | 0.0004 |
| FULL | h2_c25 | rrf@200 | 2140.1 (2186) | 0.9843 | 0.9843 | 0.9843 | 1.0000 | 0.9503 | 7872 | 0.0003 |

### webqsp

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 92.0 (134) | 0.5576 | 0.5666 | 0.6290 | 0.7352 | 0.4684 | 1738 | 0.0276 |
| STRUCT | h1_c25 | rrf@200 | 237.2 (277) | 0.6332 | 0.6483 | 0.7142 | 0.8144 | 0.5343 | 1518 | 0.0272 |
| STRUCT | h1_c100 | rrf@50 | 137.3 (267) | 0.6424 | 0.6604 | 0.7246 | 0.8071 | 0.5695 | 2616 | 0.0199 |
| STRUCT | h1_c100 | rrf@200 | 275.1 (403) | 0.7017 | 0.7252 | 0.7915 | 0.8696 | 0.6214 | 2319 | 0.0206 |
| STRUCT | h2_c25 | rrf@50 | 578.2 (1085) | 0.6485 | 0.6688 | 0.7315 | 0.8197 | 0.5536 | 2648 | 0.0033 |
| STRUCT | h2_c25 | rrf@200 | 717.0 (1221) | 0.6992 | 0.7246 | 0.7887 | 0.8716 | 0.5988 | 2272 | 0.0029 |
| NER | h1_c25 | rrf@50 | 62.2 (89) | 0.2485 | 0.2508 | 0.2803 | 0.3733 | 0.1949 | 235 | 0.0128 |
| NER | h1_c25 | rrf@200 | 209.9 (233) | 0.3875 | 0.3952 | 0.4371 | 0.5449 | 0.3041 | 207 | 0.0139 |
| NER | h1_c100 | rrf@50 | 62.5 (91) | 0.2485 | 0.2508 | 0.2803 | 0.3733 | 0.1949 | 235 | 0.0125 |
| NER | h1_c100 | rrf@200 | 210.2 (237) | 0.3875 | 0.3952 | 0.4371 | 0.5449 | 0.3041 | 207 | 0.0135 |
| NER | h2_c25 | rrf@50 | 83.1 (177) | 0.2532 | 0.2554 | 0.2856 | 0.3786 | 0.1996 | 252 | 0.0051 |
| NER | h2_c25 | rrf@200 | 230.2 (319) | 0.3907 | 0.3985 | 0.4407 | 0.5482 | 0.3067 | 219 | 0.0048 |
| KNN | h1_c25 | rrf@50 | 69.7 (91) | 0.2027 | 0.2049 | 0.2286 | 0.3134 | 0.1544 | 136 | 0.0046 |
| KNN | h1_c25 | rrf@200 | 211.5 (227) | 0.3270 | 0.3348 | 0.3688 | 0.4704 | 0.2495 | 61 | 0.0035 |
| KNN | h1_c100 | rrf@50 | 71.6 (96) | 0.2027 | 0.2049 | 0.2286 | 0.3134 | 0.1544 | 136 | 0.0042 |
| KNN | h1_c100 | rrf@200 | 212.7 (231) | 0.3270 | 0.3348 | 0.3688 | 0.4704 | 0.2495 | 61 | 0.0032 |
| KNN | h2_c25 | rrf@50 | 142.7 (226) | 0.2256 | 0.2289 | 0.2544 | 0.3453 | 0.1697 | 309 | 0.0022 |
| KNN | h2_c25 | rrf@200 | 272.2 (350) | 0.3373 | 0.3460 | 0.3805 | 0.4877 | 0.2548 | 153 | 0.0014 |
| FULL | h1_c25 | rrf@50 | 119.0 (173) | 0.5723 | 0.5815 | 0.6455 | 0.7545 | 0.4817 | 1803 | 0.0174 |
| FULL | h1_c25 | rrf@200 | 255.8 (304) | 0.6373 | 0.6525 | 0.7188 | 0.8190 | 0.5376 | 1541 | 0.0184 |
| FULL | h1_c100 | rrf@50 | 165.3 (299) | 0.6555 | 0.6737 | 0.7394 | 0.8244 | 0.5808 | 2673 | 0.0154 |
| FULL | h1_c100 | rrf@200 | 294.7 (425) | 0.7057 | 0.7294 | 0.7961 | 0.8743 | 0.6248 | 2342 | 0.0165 |
| FULL | h2_c25 | rrf@50 | 1064.0 (1843) | 0.7073 | 0.7407 | 0.7978 | 0.8703 | 0.6234 | 3505 | 0.0023 |
| FULL | h2_c25 | rrf@200 | 1175.3 (1939) | 0.7409 | 0.7781 | 0.8358 | 0.9029 | 0.6540 | 3023 | 0.0021 |

## Knee reading, beside the SOTA exposure target

Rule (declared, not chosen here): the smallest pool whose fraction of attainable recall@5 is at least 0.90, per dataset and regime. The SOTA exposure targets are quoted from the amendment; the SOTA-matched substrate per dataset is the regime the comparison systems actually had. Match, do not exceed.

| dataset | regime | knee (smallest pool reaching the rule) | gain over its base pool @5 (knee cell, else best) | candidates ÷ base | best cell in regime: ceiling@5 (fraction of attainable) at candidates | note |
|---|---|---|---|---|---|---|
| metaqa | RETRIEVAL | not reached in the declared pools | – | – | frozen_union: 0.1240 (0.150) at 378 cand. |  |
| metaqa | STRUCT | not reached in the declared pools | +0.6480 over rrf@200 | 3.81× | equal_rrf_budget_200+STRUCT:h2_c25: 0.7136 (0.862) at 762 cand. | SOTA-matched substrate |
| metaqa | NER | not reached in the declared pools | +0.0018 over rrf@200 | 1.01× | equal_rrf_budget_200+NER:h2_c25: 0.0674 (0.081) at 202 cand. |  |
| metaqa | KNN | not reached in the declared pools | +0.0142 over rrf@200 | 1.57× | equal_rrf_budget_200+KNN:h2_c25: 0.0798 (0.096) at 314 cand. |  |
| metaqa | FULL | equal_rrf_budget_200+FULL:h2_c25 — 1290 cand., ceiling@5 0.7469 (0.902 of attainable) | +0.6812 over rrf@200 | 6.45× | equal_rrf_budget_200+FULL:h2_c25: 0.7469 (0.902) at 1290 cand. |  |
| squad | RETRIEVAL | equal_rrf_budget_50 — 50 cand., ceiling@5 0.9798 (0.980 of attainable) | – | – | frozen_union: 0.9955 (0.995) at 322 cand. |  |
| squad | STRUCT | equal_rrf_budget_50+STRUCT:h1_c25 — 131 cand., ceiling@5 0.9804 (0.980 of attainable) | +0.0006 over rrf@50 | 2.63× | equal_rrf_budget_200+STRUCT:h2_c25: 0.9936 (0.994) at 535 cand. |  |
| squad | NER | equal_rrf_budget_50+NER:h1_c25 — 113 cand., ceiling@5 0.9821 (0.982 of attainable) | +0.0024 over rrf@50 | 2.26× | equal_rrf_budget_200+NER:h2_c25: 0.9943 (0.994) at 860 cand. |  |
| squad | KNN | equal_rrf_budget_50+KNN:h1_c25 — 64 cand., ceiling@5 0.9810 (0.981 of attainable) | +0.0012 over rrf@50 | 1.28× | equal_rrf_budget_200+KNN:h2_c25: 0.9941 (0.994) at 240 cand. |  |
| squad | FULL | equal_rrf_budget_50+FULL:h1_c25 — 201 cand., ceiling@5 0.9834 (0.983 of attainable) | +0.0036 over rrf@50 | 4.01× | equal_rrf_budget_200+FULL:h2_c25: 0.9954 (0.995) at 1799 cand. |  |
| musique | RETRIEVAL | not reached in the declared pools | – | – | frozen_union: 0.8131 (0.813) at 341 cand. |  |
| musique | STRUCT | not reached in the declared pools | +0.0686 over rrf@200 | 4.04× | equal_rrf_budget_200+STRUCT:h2_c25: 0.8495 (0.850) at 809 cand. |  |
| musique | NER | not reached in the declared pools | +0.1034 over rrf@200 | 5.56× | equal_rrf_budget_200+NER:h2_c25: 0.8844 (0.884) at 1112 cand. | SOTA-matched substrate |
| musique | KNN | not reached in the declared pools | +0.0471 over rrf@200 | 1.49× | equal_rrf_budget_200+KNN:h2_c25: 0.8281 (0.828) at 297 cand. |  |
| musique | FULL | equal_rrf_budget_200+FULL:h2_c25 — 2092 cand., ceiling@5 0.9207 (0.921 of attainable) | +0.1398 over rrf@200 | 10.46× | equal_rrf_budget_200+FULL:h2_c25: 0.9207 (0.921) at 2092 cand. |  |
| hotpotqa | RETRIEVAL | not reached in the declared pools | – | – | frozen_union: 0.8940 (0.894) at 354 cand. |  |
| hotpotqa | STRUCT | equal_rrf_budget_50+STRUCT:h1_c25 — 94 cand., ceiling@5 0.9800 (0.980 of attainable) | +0.1497 over rrf@50 | 1.87× | equal_rrf_budget_200+STRUCT:h2_c25: 0.9890 (0.989) at 771 cand. |  |
| hotpotqa | NER | equal_rrf_budget_200+NER:h1_c25 — 270 cand., ceiling@5 0.9018 (0.902 of attainable) | +0.0247 over rrf@200 | 1.35× | equal_rrf_budget_200+NER:h2_c25: 0.9098 (0.910) at 913 cand. | SOTA-matched substrate |
| hotpotqa | KNN | equal_rrf_budget_200+KNN:h2_c25 — 331 cand., ceiling@5 0.9027 (0.903 of attainable) | +0.0257 over rrf@200 | 1.65× | equal_rrf_budget_200+KNN:h2_c25: 0.9027 (0.903) at 331 cand. |  |
| hotpotqa | FULL | equal_rrf_budget_50+FULL:h1_c25 — 179 cand., ceiling@5 0.9822 (0.982 of attainable) | +0.1519 over rrf@50 | 3.58× | equal_rrf_budget_200+FULL:h2_c25: 0.9918 (0.992) at 1919 cand. |  |
| 2wiki | RETRIEVAL | not reached in the declared pools | – | – | frozen_union: 0.7698 (0.770) at 364 cand. |  |
| 2wiki | STRUCT | equal_rrf_budget_50+STRUCT:h1_c25 — 106 cand., ceiling@5 0.9636 (0.964 of attainable) | +0.2382 over rrf@50 | 2.11× | equal_rrf_budget_200+STRUCT:h2_c25: 0.9755 (0.975) at 1150 cand. |  |
| 2wiki | NER | not reached in the declared pools | +0.1119 over rrf@200 | 7.86× | equal_rrf_budget_200+NER:h2_c25: 0.8682 (0.868) at 1572 cand. | SOTA-matched substrate |
| 2wiki | KNN | not reached in the declared pools | +0.0498 over rrf@200 | 1.71× | equal_rrf_budget_200+KNN:h2_c25: 0.8061 (0.806) at 342 cand. |  |
| 2wiki | FULL | equal_rrf_budget_50+FULL:h1_c25 — 222 cand., ceiling@5 0.9726 (0.973 of attainable) | +0.2472 over rrf@50 | 4.43× | equal_rrf_budget_200+FULL:h2_c25: 0.9843 (0.984) at 2140 cand. |  |
| webqsp | RETRIEVAL | not reached in the declared pools | – | – | frozen_union: 0.3655 (0.412) at 327 cand. |  |
| webqsp | STRUCT | not reached in the declared pools | +0.3861 over rrf@200 | 1.38× | equal_rrf_budget_200+STRUCT:h1_c100: 0.7017 (0.792) at 275 cand. | SOTA-matched substrate |
| webqsp | NER | not reached in the declared pools | +0.0750 over rrf@200 | 1.15× | equal_rrf_budget_200+NER:h2_c25: 0.3907 (0.441) at 230 cand. |  |
| webqsp | KNN | not reached in the declared pools | +0.0217 over rrf@200 | 1.36× | equal_rrf_budget_200+KNN:h2_c25: 0.3373 (0.380) at 272 cand. |  |
| webqsp | FULL | not reached in the declared pools | +0.4253 over rrf@200 | 5.88× | equal_rrf_budget_200+FULL:h2_c25: 0.7409 (0.836) at 1175 cand. |  |

### Exposure match

What the comparison systems were given, beside what our inference-safe protocol gives on the same substrate. The KB-QA exposure is measured on our STRUCT graph from the assigned topic entities (a diagnostic; never our input); the GraphER exposure is a 200-candidate scope over a query-induced corpus, which the freeze forbids us to build, so our counterpart is the matched-substrate cell nearest 200 candidates over the full corpus.

| dataset | SOTA lineage | matched substrate | the exposure the SOTA had | ours nearest 200 candidates on that substrate: pool_ceiling@5 (of attainable) | ours best on that substrate | ours best on FULL (three families; exceeds the matched substrate) |
|---|---|---|---|---|---|---|
| metaqa | KB-QA (NuTrea / ReaRev / GNN-RAG): assigned topic entity, subgraph of as many hops as the question (1-3) | STRUCT | 2-hop topic-entity subgraph on our STRUCT graph: gold coverage 0.685 (all golds 0.658) at 1,797 nodes mean, p50 577 | `equal_rrf_budget_200+STRUCT:h1_c25`: 0.3975 (0.480) at 253 cand. | `equal_rrf_budget_200+STRUCT:h2_c25`: 0.7136 (0.862) at 762 cand. | `equal_rrf_budget_200+FULL:h2_c25`: 0.7469 (0.902) at 1290 cand. |
| squad | none (no graph-retrieval SOTA) | none | – | `equal_rrf_budget_200`: 0.9933 at 200 cand. | `frozen_union`: 0.9955 (0.995) at 322 cand. | `equal_rrf_budget_200+FULL:h2_c25`: 0.9954 (0.995) at 1799 cand. |
| musique | GraphER: 200-candidate scope over a query-induced corpus, PR@K | NER | 200 candidates per query over a corpus induced from 2,000 sampled dev queries (corpus subsetting by query -- forbidden for us) | `equal_rrf_budget_50+NER:h1_c100`: 0.8055 (0.806) at 168 cand. | `equal_rrf_budget_200+NER:h2_c25`: 0.8844 (0.884) at 1112 cand. | `equal_rrf_budget_200+FULL:h2_c25`: 0.9207 (0.921) at 2092 cand. |
| hotpotqa | GraphER: 200-candidate scope over a query-induced corpus (2,000 sampled dev queries), PR@K | NER | 200 candidates per query over a corpus induced from 2,000 sampled dev queries (corpus subsetting by query -- forbidden for us) | `equal_rrf_budget_50+NER:h1_c100`: 0.8679 (0.868) at 148 cand. | `equal_rrf_budget_200+NER:h2_c25`: 0.9098 (0.910) at 913 cand. | `equal_rrf_budget_200+FULL:h2_c25`: 0.9918 (0.992) at 1919 cand. |
| 2wiki | GraphER: 200-candidate scope over a query-induced corpus, PR@K | NER | 200 candidates per query over a corpus induced from 2,000 sampled dev queries (corpus subsetting by query -- forbidden for us) | `equal_rrf_budget_50+NER:h1_c100`: 0.8365 (0.837) at 214 cand. | `equal_rrf_budget_200+NER:h2_c25`: 0.8682 (0.868) at 1572 cand. | `equal_rrf_budget_200+FULL:h2_c25`: 0.9843 (0.984) at 2140 cand. |
| webqsp | KB-QA (NuTrea / ReaRev / GNN-RAG): assigned topic entities, 2-hop subgraph | STRUCT | 2-hop topic-entity subgraph on our STRUCT graph: gold coverage 0.989 (all golds 0.969) at 161,690 nodes mean, p50 169,768 | `equal_rrf_budget_200+STRUCT:h1_c25`: 0.6332 (0.714) at 237 cand. | `equal_rrf_budget_200+STRUCT:h1_c100`: 0.7017 (0.792) at 275 cand. | `equal_rrf_budget_200+FULL:h2_c25`: 0.7409 (0.836) at 1175 cand. |

**SOTA exposure targets (amendment 1):**

- `metaqa_webqsp`: The KB systems assign at least one topic entity per question (an oracle under the archaeology oracle_rule) and extract a bounded subgraph around it; the answer-in-subgraph rate is the exposure they had. We refuse assigned topic entities; our seeds are retrieved. Target: the reported subgraph answer coverage where a paper states it, otherwise the bounded 2-hop exposure from retrieved seeds, labelled as the inference-safe counterpart.
- `hotpotqa_2wiki_musique`: GraphER reranks a 200-candidate scope drawn from a corpus induced from 2,000 sampled dev queries -- corpus subsetting by query, which the freeze forbids for us -- and reports PR@K, a set-coverage metric. Target: our frozen_union / equal_rrf_budget_200 over the full corpus, with the corpus difference stated on the row.
- `squad`: no graph-retrieval SOTA exists; retrieval pools only.

## Readings (per dataset; ceilings, not results)

Each paragraph names the cells behind it. `attainable` is `recall_ceiling_perfect_retrieval@5`, the most a K=5 cut-off can return given the gold count; `of attainable` is the fraction of that the pool reaches.

**metaqa** (`dev`, 39,138 queries scored; 7.69 golds per query, attainable@5 0.8279). Retrieval alone: the best inherited pool is `frozen_union` at pool_ceiling@5 0.1240 (0.150 of attainable, 378 candidates); `dense_top200` 0.0669, `equal_rrf_budget_200` 0.0656. Best declared graph cell: `equal_rrf_budget_200+FULL:h2_c25` 0.7469 (0.902 of attainable) at 1290 candidates, 152,436 missing golds recovered over its base. On the SOTA-matched substrate (STRUCT): best cell `equal_rrf_budget_200+STRUCT:h2_c25` 0.7136 (0.862 of attainable) at 762 candidates. Of the 279,488 golds the `frozen_union` pool misses (36,160 of 39,138 walked queries, eval), 0.486 lie within 2 undirected STRUCT hops of the seeds and 0.996 within 3; on FULL 0.551 and 0.998. Oracle topic-entity exposure (diagnostic): within 1 STRUCT hop of the assigned topic entities gold coverage is 0.301 (mean subgraph 8 nodes, p50 7); within 2 hops 0.685 (mean 1,797 nodes, p50 577, p95 5,324). That is the exposure the KB-QA lineage had; our seeds are retrieved, and the pools above never use it.

**squad** (`dev`, 11,873 queries scored; 1.00 golds per query, attainable@5 1.0000). Retrieval alone: the best inherited pool is `frozen_union` at pool_ceiling@5 0.9955 (0.995 of attainable, 322 candidates); `dense_top200` 0.9880, `equal_rrf_budget_200` 0.9933. Best declared graph cell: `equal_rrf_budget_200+FULL:h2_c25` 0.9954 (0.995 of attainable) at 1799 candidates, 25 missing golds recovered over its base. No graph-retrieval SOTA exists for this dataset; the retrieval pools are the comparison. Of the 54 golds the `frozen_union` pool misses (54 of 11,873 walked queries, eval), 0.519 lie within 2 undirected STRUCT hops of the seeds and 0.741 within 3; on FULL 0.722 and 0.981.

**musique** (`dev`, 2,417 queries scored; 2.65 golds per query, attainable@5 1.0000). Retrieval alone: the best inherited pool is `frozen_union` at pool_ceiling@5 0.8131 (0.813 of attainable, 341 candidates); `dense_top200` 0.7788, `equal_rrf_budget_200` 0.7809. Best declared graph cell: `equal_rrf_budget_200+FULL:h2_c25` 0.9207 (0.921 of attainable) at 2092 candidates, 920 missing golds recovered over its base. On the SOTA-matched substrate (NER): best cell `equal_rrf_budget_200+NER:h2_c25` 0.8844 (0.884 of attainable) at 1112 candidates. Of the 1,311 golds the `frozen_union` pool misses (1,003 of 2,417 walked queries, eval), 0.770 lie within 2 undirected STRUCT hops of the seeds and 0.935 within 3; on FULL 0.892 and 1.000.

**hotpotqa** (`validation`, 7,405 queries scored; 2.00 golds per query, attainable@5 1.0000). Retrieval alone: the best inherited pool is `frozen_union` at pool_ceiling@5 0.8940 (0.894 of attainable, 354 candidates); `dense_top200` 0.8531, `equal_rrf_budget_200` 0.8770. Best declared graph cell: `equal_rrf_budget_200+FULL:h2_c25` 0.9918 (0.992 of attainable) at 1919 candidates, 1,699 missing golds recovered over its base. On the SOTA-matched substrate (NER): best cell `equal_rrf_budget_200+NER:h2_c25` 0.9098 (0.910 of attainable) at 913 candidates. Of the 385 golds the `frozen_union` pool misses (378 of 1,852 walked queries, eval every 4th query), 0.945 lie within 2 undirected STRUCT hops of the seeds and 0.977 within 3; on FULL 0.963 and 0.992.

**2wiki** (`dev`, 12,576 queries scored; 2.44 golds per query, attainable@5 1.0000). Retrieval alone: the best inherited pool is `frozen_union` at pool_ceiling@5 0.7698 (0.770 of attainable, 364 candidates); `dense_top200` 0.7195, `equal_rrf_budget_200` 0.7563. Best declared graph cell: `equal_rrf_budget_200+FULL:h2_c25` 0.9843 (0.984 of attainable) at 2140 candidates, 7,872 missing golds recovered over its base. On the SOTA-matched substrate (NER): best cell `equal_rrf_budget_200+NER:h2_c25` 0.8682 (0.868 of attainable) at 1572 candidates. Of the 1,381 golds the `frozen_union` pool misses (1,038 of 2,096 walked queries, eval every 6th query), 0.930 lie within 2 undirected STRUCT hops of the seeds and 0.987 within 3; on FULL 0.982 and 0.998.

**webqsp** (`train_holdout`, 1,503 queries scored; 7.04 golds per query, attainable@5 0.8865). Retrieval alone: the best inherited pool is `frozen_union` at pool_ceiling@5 0.3655 (0.412 of attainable, 327 candidates); `dense_top200` 0.3576, `equal_rrf_budget_200` 0.3156. Best declared graph cell: `equal_rrf_budget_200+FULL:h2_c25` 0.7409 (0.836 of attainable) at 1175 candidates, 3,023 missing golds recovered over its base. On the SOTA-matched substrate (STRUCT): best cell `equal_rrf_budget_200+STRUCT:h1_c100` 0.7017 (0.792 of attainable) at 275 candidates. Of the 8,704 golds the `frozen_union` pool misses (1,074 of 1,503 walked queries, eval), 0.898 lie within 2 undirected STRUCT hops of the seeds and 0.989 within 3; on FULL 0.927 and 0.996. Oracle topic-entity exposure (diagnostic): within 1 STRUCT hop of the assigned topic entities gold coverage is 0.653 (mean subgraph 1,920 nodes, p50 252); within 2 hops 0.989 (mean 161,690 nodes, p50 169,768, p95 280,435). That is the exposure the KB-QA lineage had; our seeds are retrieved, and the pools above never use it.


## Reachability of the missing gold (inherited diagnostic)

For the gold nodes the `frozen_union` pool misses, how many undirected hops from the seeds they sit, per regime graph, up to three hops and two million visited nodes per query. Strided populations are labelled. A missing gold beyond three hops is not reachable by any bounded expansion from these seeds.


### metaqa

| regime (undirected) | population | queries walked | with missing gold | missing golds | within 1 hop | within 2 | within 3 | beyond 3 / unreachable | frontier-capped queries | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | eval | 39,138 | 36,160 | 279,488 | 0.1182 | 0.4864 | 0.9962 | 0.0038 | 0 | 131.6 |
| NER | eval | 39,138 | 36,160 | 279,488 | 0.0003 | 0.0003 | 0.0003 | 0.9997 | 0 | 7.2 |
| KNN | eval | 39,138 | 36,160 | 279,488 | 0.0015 | 0.0059 | 0.0174 | 0.9826 | 0 | 19.2 |
| FULL | eval | 39,138 | 36,160 | 279,488 | 0.1195 | 0.5508 | 0.9975 | 0.0025 | 0 | 135.6 |

### squad

| regime (undirected) | population | queries walked | with missing gold | missing golds | within 1 hop | within 2 | within 3 | beyond 3 / unreachable | frontier-capped queries | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | eval | 11,873 | 54 | 54 | 0.0741 | 0.5185 | 0.7407 | 0.2593 | 0 | 3.9 |
| NER | eval | 11,873 | 54 | 54 | 0.0370 | 0.2222 | 0.6852 | 0.3148 | 0 | 2.5 |
| KNN | eval | 11,873 | 54 | 54 | 0.0000 | 0.0741 | 0.0926 | 0.9074 | 0 | 3.9 |
| FULL | eval | 11,873 | 54 | 54 | 0.1111 | 0.7222 | 0.9815 | 0.0185 | 0 | 5.8 |

### musique

| regime (undirected) | population | queries walked | with missing gold | missing golds | within 1 hop | within 2 | within 3 | beyond 3 / unreachable | frontier-capped queries | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | eval | 2,417 | 1,003 | 1,311 | 0.2624 | 0.7696 | 0.9352 | 0.0648 | 0 | 52.6 |
| NER | eval | 2,417 | 1,003 | 1,311 | 0.3471 | 0.4851 | 0.6270 | 0.3730 | 0 | 4.6 |
| KNN | eval | 2,417 | 1,003 | 1,311 | 0.1106 | 0.1777 | 0.2563 | 0.7437 | 0 | 0.7 |
| FULL | eval | 2,417 | 1,003 | 1,311 | 0.4638 | 0.8917 | 1.0000 | 0.0000 | 0 | 38.3 |

### hotpotqa

| regime (undirected) | population | queries walked | with missing gold | missing golds | within 1 hop | within 2 | within 3 | beyond 3 / unreachable | frontier-capped queries | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | eval every 4th query | 1,852 | 378 | 385 | 0.9143 | 0.9455 | 0.9766 | 0.0234 | 0 | 15.8 |
| NER | eval every 4th query | 1,852 | 378 | 385 | 0.2026 | 0.2779 | 0.3636 | 0.6364 | 0 | 2.2 |
| KNN | eval every 4th query | 1,852 | 378 | 385 | 0.1013 | 0.1844 | 0.2519 | 0.7481 | 0 | 0.3 |
| FULL | eval every 4th query | 1,852 | 378 | 385 | 0.9267 | 0.9634 | 0.9921 | 0.0079 | 2 | 21.3 |

### 2wiki

| regime (undirected) | population | queries walked | with missing gold | missing golds | within 1 hop | within 2 | within 3 | beyond 3 / unreachable | frontier-capped queries | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | eval every 6th query | 2,096 | 1,038 | 1,381 | 0.8776 | 0.9299 | 0.9875 | 0.0125 | 15 | 175.1 |
| NER | eval every 6th query | 2,096 | 1,038 | 1,381 | 0.3519 | 0.4453 | 0.5677 | 0.4323 | 0 | 38.7 |
| KNN | eval every 6th query | 2,096 | 1,038 | 1,381 | 0.0891 | 0.1644 | 0.2129 | 0.7871 | 0 | 0.5 |
| FULL | eval every 6th query | 2,096 | 1,038 | 1,381 | 0.9213 | 0.9817 | 0.9985 | 0.0015 | 45 | 157.6 |

### webqsp

| regime (undirected) | population | queries walked | with missing gold | missing golds | within 1 hop | within 2 | within 3 | beyond 3 / unreachable | frontier-capped queries | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | eval | 1,503 | 1,074 | 8,704 | 0.3655 | 0.8976 | 0.9893 | 0.0107 | 2 | 158.4 |
| NER | eval | 1,503 | 1,074 | 8,704 | 0.0222 | 0.0236 | 0.0249 | 0.9751 | 0 | 0.9 |
| KNN | eval | 1,503 | 1,074 | 8,704 | 0.0052 | 0.0138 | 0.0323 | 0.9677 | 0 | 1.1 |
| FULL | eval | 1,503 | 1,074 | 8,704 | 0.3636 | 0.9274 | 0.9962 | 0.0038 | 8 | 126.9 |

## Oracle topic-entity exposure (diagnostic column only)


### metaqa

Status: **DIAGNOSTIC_COLUMN_ONLY** — the exposure the KB-QA lineage had, measured on our served STRUCT graph; never a seed, never a pool.

| setting | value |
|---|---|
| graph | structural, undirected |
| visited_cap | 2000000 |
| queries_without_topic_entity | 0 |
| queries_capped | 0 |

| hops from the assigned topic entities | queries | gold coverage (reference level, macro) | queries with any gold covered | queries with all gold covered | subgraph nodes mean | p50 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 39,138 | 0.3011 | 0.4111 | 0.2772 | 8 | 7 | 18 | 518 |
| 2 | 39,138 | 0.6848 | 0.8018 | 0.6581 | 1,797 | 577 | 5,324 | 9,648 |

### webqsp

Status: **DIAGNOSTIC_COLUMN_ONLY** — the exposure the KB-QA lineage had, measured on our served STRUCT graph; never a seed, never a pool.

| setting | value |
|---|---|
| graph | structural, undirected |
| visited_cap | 2000000 |
| queries_without_topic_entity | 1 |
| queries_capped | 0 |

| hops from the assigned topic entities | queries | gold coverage (reference level, macro) | queries with any gold covered | queries with all gold covered | subgraph nodes mean | p50 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,502 | 0.6530 | 0.6897 | 0.6305 | 1,920 | 252 | 4,871 | 130,076 |
| 2 | 1,502 | 0.9891 | 0.9940 | 0.9687 | 161,690 | 169,768 | 280,435 | 1,328,520 |

## LEGACY_CONTINUITY paired column (NOT an evaluation population)

The legacy 2,000-query sets mapped to canonical ids: squad/musique/2wiki are TRAIN questions, hotpotqa is 1,842 train + 158 validation, metaqa is 1,998 dev with 46 gold disagreements. Reported only so a reader can place the M0A–M2D numbers next to the served substrate; never averaged with the rows above.


### metaqa — 1,998 legacy queries

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.0593 | 0.0709 | 0.0722 | 0.0844 | 0.1787 | 0.0415 | 0.0415 |
| splade_top200 | 200.0 (200) | 0.0570 | 0.0664 | 0.0665 | 0.0790 | 0.1181 | 0.0475 | 0.0475 |
| frozen_union | 377.1 (396) | 0.1076 | 0.1283 | 0.1296 | 0.1525 | 0.2728 | 0.0821 | 0.0821 |
| equal_rrf_budget_50 | 50.0 (50) | 0.0239 | 0.0270 | 0.0270 | 0.0321 | 0.0806 | 0.0165 | 0.0165 |
| equal_rrf_budget_100 | 100.0 (100) | 0.0351 | 0.0405 | 0.0406 | 0.0481 | 0.1121 | 0.0245 | 0.0245 |
| equal_rrf_budget_200 | 200.0 (200) | 0.0574 | 0.0661 | 0.0666 | 0.0787 | 0.1667 | 0.0395 | 0.0395 |
| equal_rrf_budget_400 | 377.1 (396) | 0.1076 | 0.1283 | 0.1296 | 0.1525 | 0.2728 | 0.0821 | 0.0821 |

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 103.7 (138) | 0.4427 | 0.4482 | 0.5265 | 0.6291 | 0.3894 | 2080 | 0.0194 |
| STRUCT | h1_c25 | rrf@200 | 253.0 (287) | 0.4612 | 0.4672 | 0.5484 | 0.6702 | 0.3984 | 1988 | 0.0188 |
| STRUCT | h1_c100 | rrf@50 | 105.2 (145) | 0.4432 | 0.4487 | 0.5271 | 0.6291 | 0.3909 | 2109 | 0.0191 |
| STRUCT | h1_c100 | rrf@200 | 254.5 (293) | 0.4617 | 0.4678 | 0.5490 | 0.6702 | 0.3999 | 2016 | 0.0185 |
| STRUCT | h2_c25 | rrf@50 | 621.1 (953) | 0.7350 | 0.7833 | 0.8741 | 0.9244 | 0.7027 | 6521 | 0.0057 |
| STRUCT | h2_c25 | rrf@200 | 764.3 (1092) | 0.7393 | 0.7881 | 0.8792 | 0.9354 | 0.7047 | 6297 | 0.0056 |
| NER | h1_c25 | rrf@50 | 51.7 (56) | 0.0290 | 0.0290 | 0.0345 | 0.0831 | 0.0180 | 7 | 0.0021 |
| NER | h1_c25 | rrf@200 | 201.4 (205) | 0.0674 | 0.0679 | 0.0802 | 0.1682 | 0.0405 | 4 | 0.0015 |
| NER | h1_c100 | rrf@50 | 51.7 (56) | 0.0290 | 0.0290 | 0.0345 | 0.0831 | 0.0180 | 7 | 0.0021 |
| NER | h1_c100 | rrf@200 | 201.4 (205) | 0.0674 | 0.0679 | 0.0802 | 0.1682 | 0.0405 | 4 | 0.0015 |
| NER | h2_c25 | rrf@50 | 52.1 (59) | 0.0290 | 0.0290 | 0.0345 | 0.0831 | 0.0180 | 7 | 0.0017 |
| NER | h2_c25 | rrf@200 | 201.8 (208) | 0.0674 | 0.0679 | 0.0802 | 0.1682 | 0.0405 | 4 | 0.0011 |
| KNN | h1_c25 | rrf@50 | 75.4 (94) | 0.0327 | 0.0327 | 0.0389 | 0.0946 | 0.0195 | 45 | 0.0009 |
| KNN | h1_c25 | rrf@200 | 218.5 (236) | 0.0694 | 0.0699 | 0.0825 | 0.1737 | 0.0415 | 24 | 0.0006 |
| KNN | h1_c100 | rrf@50 | 75.7 (97) | 0.0327 | 0.0327 | 0.0389 | 0.0946 | 0.0195 | 45 | 0.0009 |
| KNN | h1_c100 | rrf@200 | 218.8 (238) | 0.0694 | 0.0699 | 0.0825 | 0.1737 | 0.0415 | 24 | 0.0006 |
| KNN | h2_c25 | rrf@50 | 179.2 (268) | 0.0477 | 0.0477 | 0.0567 | 0.1361 | 0.0260 | 182 | 0.0007 |
| KNN | h2_c25 | rrf@200 | 311.7 (394) | 0.0783 | 0.0789 | 0.0931 | 0.1987 | 0.0455 | 116 | 0.0005 |
| FULL | h1_c25 | rrf@50 | 129.7 (171) | 0.4457 | 0.4511 | 0.5301 | 0.6366 | 0.3909 | 2114 | 0.0133 |
| FULL | h1_c25 | rrf@200 | 272.1 (312) | 0.4630 | 0.4691 | 0.5506 | 0.6752 | 0.3994 | 2008 | 0.0139 |
| FULL | h1_c100 | rrf@50 | 131.5 (177) | 0.4462 | 0.4517 | 0.5306 | 0.6366 | 0.3924 | 2143 | 0.0132 |
| FULL | h1_c100 | rrf@200 | 273.8 (318) | 0.4635 | 0.4696 | 0.5512 | 0.6752 | 0.4009 | 2036 | 0.0138 |
| FULL | h2_c25 | rrf@50 | 1165.7 (1712) | 0.7668 | 0.8283 | 0.9119 | 0.9550 | 0.7277 | 7538 | 0.0034 |
| FULL | h2_c25 | rrf@200 | 1289.3 (1832) | 0.7700 | 0.8321 | 0.9157 | 0.9615 | 0.7292 | 7280 | 0.0033 |

### squad — 2,000 legacy queries

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.9915 | 0.9915 | 0.9915 | 0.9915 | 0.9915 | 0.9915 | 0.9915 |
| splade_top200 | 200.0 (200) | 0.9865 | 0.9865 | 0.9865 | 0.9865 | 0.9865 | 0.9865 | 0.9865 |
| frozen_union | 317.0 (364) | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9960 |
| equal_rrf_budget_50 | 50.0 (50) | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 |
| equal_rrf_budget_100 | 100.0 (100) | 0.9885 | 0.9885 | 0.9885 | 0.9885 | 0.9885 | 0.9885 | 0.9885 |
| equal_rrf_budget_200 | 200.0 (200) | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9940 |
| equal_rrf_budget_400 | 317.0 (364) | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9960 |

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 128.5 (199) | 0.9820 | 0.9820 | 0.9820 | 0.9820 | 0.9820 | 2 | 0.0000 |
| STRUCT | h1_c25 | rrf@200 | 272.8 (343) | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0 | 0.0000 |
| STRUCT | h1_c100 | rrf@50 | 257.1 (471) | 0.9820 | 0.9820 | 0.9820 | 0.9820 | 0.9820 | 2 | 0.0000 |
| STRUCT | h1_c100 | rrf@200 | 392.6 (605) | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0 | 0.0000 |
| STRUCT | h2_c25 | rrf@50 | 378.7 (706) | 0.9820 | 0.9820 | 0.9820 | 0.9820 | 0.9820 | 2 | 0.0000 |
| STRUCT | h2_c25 | rrf@200 | 517.5 (839) | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0 | 0.0000 |
| NER | h1_c25 | rrf@50 | 115.6 (182) | 0.9835 | 0.9835 | 0.9835 | 0.9835 | 0.9835 | 5 | 0.0000 |
| NER | h1_c25 | rrf@200 | 255.3 (317) | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 2 | 0.0000 |
| NER | h1_c100 | rrf@50 | 137.7 (251) | 0.9840 | 0.9840 | 0.9840 | 0.9840 | 0.9840 | 6 | 0.0000 |
| NER | h1_c100 | rrf@200 | 275.5 (379) | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 2 | 0.0000 |
| NER | h2_c25 | rrf@50 | 769.8 (1475) | 0.9850 | 0.9850 | 0.9850 | 0.9850 | 0.9850 | 8 | 0.0000 |
| NER | h2_c25 | rrf@200 | 886.6 (1573) | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 3 | 0.0000 |
| KNN | h1_c25 | rrf@50 | 63.7 (80) | 0.9830 | 0.9830 | 0.9830 | 0.9830 | 0.9830 | 4 | 0.0001 |
| KNN | h1_c25 | rrf@200 | 207.5 (222) | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 1 | 0.0001 |
| KNN | h1_c100 | rrf@50 | 63.7 (80) | 0.9830 | 0.9830 | 0.9830 | 0.9830 | 0.9830 | 4 | 0.0001 |
| KNN | h1_c100 | rrf@200 | 207.5 (222) | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 1 | 0.0001 |
| KNN | h2_c25 | rrf@50 | 106.8 (167) | 0.9840 | 0.9840 | 0.9840 | 0.9840 | 0.9840 | 6 | 0.0001 |
| KNN | h2_c25 | rrf@200 | 239.5 (298) | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 1 | 0.0000 |
| FULL | h1_c25 | rrf@50 | 199.8 (320) | 0.9855 | 0.9855 | 0.9855 | 0.9855 | 0.9855 | 9 | 0.0000 |
| FULL | h1_c25 | rrf@200 | 331.1 (447) | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 2 | 0.0000 |
| FULL | h1_c100 | rrf@50 | 343.0 (617) | 0.9860 | 0.9860 | 0.9860 | 0.9860 | 0.9860 | 10 | 0.0000 |
| FULL | h1_c100 | rrf@200 | 465.7 (732) | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 2 | 0.0000 |
| FULL | h2_c25 | rrf@50 | 1735.3 (2028) | 0.9870 | 0.9870 | 0.9870 | 0.9870 | 0.9870 | 12 | 0.0000 |
| FULL | h2_c25 | rrf@200 | 1827.1 (2144) | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 3 | 0.0000 |

### musique — 2,000 legacy queries

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.4412 | 0.8082 | 0.8082 | 0.8082 | 0.9845 | 0.6005 | 0.6005 |
| splade_top200 | 200.0 (200) | 0.4414 | 0.7619 | 0.7619 | 0.7619 | 0.9855 | 0.5055 | 0.5055 |
| frozen_union | 342.8 (381) | 0.4458 | 0.8502 | 0.8502 | 0.8502 | 0.9955 | 0.6670 | 0.6670 |
| equal_rrf_budget_50 | 50.0 (50) | 0.4388 | 0.7346 | 0.7346 | 0.7346 | 0.9790 | 0.4705 | 0.4705 |
| equal_rrf_budget_100 | 100.0 (100) | 0.4420 | 0.7831 | 0.7831 | 0.7831 | 0.9865 | 0.5510 | 0.5510 |
| equal_rrf_budget_200 | 200.0 (200) | 0.4442 | 0.8194 | 0.8194 | 0.8194 | 0.9915 | 0.6130 | 0.6130 |
| equal_rrf_budget_400 | 342.8 (381) | 0.4458 | 0.8502 | 0.8502 | 0.8502 | 0.9955 | 0.6670 | 0.6670 |

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 139.1 (202) | 0.7845 | 0.7845 | 0.7845 | 0.9825 | 0.5625 | 239 | 0.0013 |
| STRUCT | h1_c25 | rrf@200 | 284.9 (348) | 0.8523 | 0.8523 | 0.8523 | 0.9920 | 0.6810 | 160 | 0.0009 |
| STRUCT | h1_c100 | rrf@50 | 250.3 (466) | 0.8027 | 0.8027 | 0.8027 | 0.9850 | 0.5955 | 327 | 0.0008 |
| STRUCT | h1_c100 | rrf@200 | 391.3 (602) | 0.8632 | 0.8632 | 0.8632 | 0.9920 | 0.7040 | 213 | 0.0006 |
| STRUCT | h2_c25 | rrf@50 | 667.6 (1105) | 0.8050 | 0.8050 | 0.8050 | 0.9850 | 0.5980 | 337 | 0.0003 |
| STRUCT | h2_c25 | rrf@200 | 806.4 (1242) | 0.8629 | 0.8629 | 0.8629 | 0.9930 | 0.7005 | 212 | 0.0002 |
| NER | h1_c25 | rrf@50 | 135.0 (194) | 0.7947 | 0.7947 | 0.7947 | 0.9825 | 0.5785 | 303 | 0.0018 |
| NER | h1_c25 | rrf@200 | 278.3 (337) | 0.8641 | 0.8641 | 0.8641 | 0.9915 | 0.7030 | 231 | 0.0015 |
| NER | h1_c100 | rrf@50 | 161.8 (269) | 0.8013 | 0.8013 | 0.8013 | 0.9830 | 0.5890 | 334 | 0.0015 |
| NER | h1_c100 | rrf@200 | 303.8 (403) | 0.8678 | 0.8678 | 0.8678 | 0.9915 | 0.7100 | 248 | 0.0012 |
| NER | h2_c25 | rrf@50 | 948.5 (1615) | 0.8263 | 0.8263 | 0.8263 | 0.9835 | 0.6430 | 460 | 0.0003 |
| NER | h2_c25 | rrf@200 | 1078.8 (1743) | 0.8834 | 0.8834 | 0.8834 | 0.9920 | 0.7450 | 327 | 0.0002 |
| KNN | h1_c25 | rrf@50 | 75.4 (94) | 0.7622 | 0.7622 | 0.7622 | 0.9840 | 0.5175 | 136 | 0.0027 |
| KNN | h1_c25 | rrf@200 | 218.3 (237) | 0.8327 | 0.8327 | 0.8327 | 0.9920 | 0.6390 | 68 | 0.0019 |
| KNN | h1_c100 | rrf@50 | 75.4 (94) | 0.7622 | 0.7622 | 0.7622 | 0.9840 | 0.5175 | 136 | 0.0027 |
| KNN | h1_c100 | rrf@200 | 218.3 (237) | 0.8327 | 0.8327 | 0.8327 | 0.9920 | 0.6390 | 68 | 0.0019 |
| KNN | h2_c25 | rrf@50 | 169.2 (255) | 0.7914 | 0.7914 | 0.7914 | 0.9865 | 0.5710 | 276 | 0.0012 |
| KNN | h2_c25 | rrf@200 | 300.8 (389) | 0.8478 | 0.8478 | 0.8478 | 0.9925 | 0.6685 | 143 | 0.0007 |
| FULL | h1_c25 | rrf@50 | 238.9 (347) | 0.8355 | 0.8355 | 0.8355 | 0.9870 | 0.6555 | 492 | 0.0013 |
| FULL | h1_c25 | rrf@200 | 374.1 (479) | 0.8885 | 0.8885 | 0.8885 | 0.9925 | 0.7525 | 344 | 0.0010 |
| FULL | h1_c100 | rrf@50 | 370.3 (627) | 0.8520 | 0.8520 | 0.8520 | 0.9880 | 0.6870 | 570 | 0.0009 |
| FULL | h1_c100 | rrf@200 | 501.0 (754) | 0.8991 | 0.8991 | 0.8991 | 0.9925 | 0.7750 | 395 | 0.0007 |
| FULL | h2_c25 | rrf@50 | 1976.5 (2037) | 0.8858 | 0.8858 | 0.8858 | 0.9890 | 0.7540 | 736 | 0.0002 |
| FULL | h2_c25 | rrf@200 | 2087.7 (2173) | 0.9190 | 0.9190 | 0.9190 | 0.9940 | 0.8160 | 494 | 0.0001 |

### hotpotqa — 2,000 legacy queries

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.4955 | 0.8620 | 0.8620 | 0.8620 | 0.9910 | 0.7330 | 0.7330 |
| splade_top200 | 200.0 (200) | 0.4953 | 0.8455 | 0.8455 | 0.8455 | 0.9905 | 0.7005 | 0.7005 |
| frozen_union | 353.6 (389) | 0.4983 | 0.9018 | 0.9018 | 0.9018 | 0.9965 | 0.8070 | 0.8070 |
| equal_rrf_budget_50 | 50.0 (50) | 0.4948 | 0.8357 | 0.8357 | 0.8357 | 0.9895 | 0.6820 | 0.6820 |
| equal_rrf_budget_100 | 100.0 (100) | 0.4965 | 0.8662 | 0.8662 | 0.8662 | 0.9930 | 0.7395 | 0.7395 |
| equal_rrf_budget_200 | 200.0 (200) | 0.4980 | 0.8858 | 0.8858 | 0.8858 | 0.9960 | 0.7755 | 0.7755 |
| equal_rrf_budget_400 | 353.6 (389) | 0.4983 | 0.9018 | 0.9018 | 0.9018 | 0.9965 | 0.8070 | 0.8070 |

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 93.5 (138) | 0.9790 | 0.9790 | 0.9790 | 0.9930 | 0.9650 | 573 | 0.0066 |
| STRUCT | h1_c25 | rrf@200 | 238.9 (280) | 0.9882 | 0.9882 | 0.9882 | 0.9970 | 0.9795 | 410 | 0.0053 |
| STRUCT | h1_c100 | rrf@50 | 108.3 (200) | 0.9808 | 0.9808 | 0.9808 | 0.9930 | 0.9685 | 580 | 0.0050 |
| STRUCT | h1_c100 | rrf@200 | 251.4 (335) | 0.9892 | 0.9892 | 0.9892 | 0.9970 | 0.9815 | 414 | 0.0040 |
| STRUCT | h2_c25 | rrf@50 | 633.4 (1135) | 0.9845 | 0.9845 | 0.9845 | 0.9945 | 0.9745 | 595 | 0.0005 |
| STRUCT | h2_c25 | rrf@200 | 770.2 (1265) | 0.9902 | 0.9902 | 0.9902 | 0.9970 | 0.9835 | 418 | 0.0004 |
| NER | h1_c25 | rrf@50 | 124.3 (195) | 0.8712 | 0.8712 | 0.8712 | 0.9920 | 0.7505 | 142 | 0.0010 |
| NER | h1_c25 | rrf@200 | 269.9 (340) | 0.9103 | 0.9103 | 0.9103 | 0.9975 | 0.8230 | 98 | 0.0007 |
| NER | h1_c100 | rrf@50 | 149.5 (284) | 0.8735 | 0.8735 | 0.8735 | 0.9925 | 0.7545 | 151 | 0.0008 |
| NER | h1_c100 | rrf@200 | 294.4 (425) | 0.9120 | 0.9120 | 0.9120 | 0.9975 | 0.8265 | 105 | 0.0006 |
| NER | h2_c25 | rrf@50 | 787.3 (1777) | 0.8828 | 0.8828 | 0.8828 | 0.9925 | 0.7730 | 188 | 0.0001 |
| NER | h2_c25 | rrf@200 | 927.5 (1915) | 0.9175 | 0.9175 | 0.9175 | 0.9975 | 0.8375 | 127 | 0.0001 |
| KNN | h1_c25 | rrf@50 | 78.4 (100) | 0.8608 | 0.8608 | 0.8608 | 0.9910 | 0.7305 | 100 | 0.0018 |
| KNN | h1_c25 | rrf@200 | 222.1 (243) | 0.9020 | 0.9020 | 0.9020 | 0.9965 | 0.8075 | 65 | 0.0015 |
| KNN | h1_c100 | rrf@50 | 78.6 (101) | 0.8608 | 0.8608 | 0.8608 | 0.9910 | 0.7305 | 100 | 0.0017 |
| KNN | h1_c100 | rrf@200 | 222.2 (244) | 0.9020 | 0.9020 | 0.9020 | 0.9965 | 0.8075 | 65 | 0.0015 |
| KNN | h2_c25 | rrf@50 | 197.0 (309) | 0.8755 | 0.8755 | 0.8755 | 0.9910 | 0.7600 | 159 | 0.0005 |
| KNN | h2_c25 | rrf@200 | 331.7 (442) | 0.9103 | 0.9103 | 0.9103 | 0.9965 | 0.8240 | 98 | 0.0004 |
| FULL | h1_c25 | rrf@50 | 179.2 (276) | 0.9815 | 0.9815 | 0.9815 | 0.9950 | 0.9680 | 583 | 0.0023 |
| FULL | h1_c25 | rrf@200 | 317.5 (413) | 0.9898 | 0.9898 | 0.9898 | 0.9985 | 0.9810 | 416 | 0.0018 |
| FULL | h1_c100 | rrf@50 | 216.1 (389) | 0.9830 | 0.9830 | 0.9830 | 0.9950 | 0.9710 | 589 | 0.0018 |
| FULL | h1_c100 | rrf@200 | 352.2 (523) | 0.9905 | 0.9905 | 0.9905 | 0.9985 | 0.9825 | 419 | 0.0014 |
| FULL | h2_c25 | rrf@50 | 1804.4 (2037) | 0.9895 | 0.9895 | 0.9895 | 0.9965 | 0.9825 | 615 | 0.0002 |
| FULL | h2_c25 | rrf@200 | 1923.9 (2178) | 0.9938 | 0.9938 | 0.9938 | 0.9990 | 0.9885 | 432 | 0.0001 |

### 2wiki — 2,000 legacy queries

| pool | candidates mean (p95) | pool_ceiling@1 | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | ALL | FullCov ceiling@20 |
|---|---|---|---|---|---|---|---|---|
| dense_top200 | 200.0 (200) | 0.4460 | 0.6847 | 0.6847 | 0.6847 | 0.9960 | 0.3725 | 0.3725 |
| splade_top200 | 200.0 (200) | 0.4476 | 0.7154 | 0.7154 | 0.7154 | 0.9995 | 0.4080 | 0.4080 |
| frozen_union | 368.7 (393) | 0.4479 | 0.7425 | 0.7425 | 0.7425 | 1.0000 | 0.4585 | 0.4585 |
| equal_rrf_budget_50 | 50.0 (50) | 0.4476 | 0.7027 | 0.7027 | 0.7027 | 0.9995 | 0.3870 | 0.3870 |
| equal_rrf_budget_100 | 100.0 (100) | 0.4479 | 0.7164 | 0.7164 | 0.7164 | 1.0000 | 0.4105 | 0.4105 |
| equal_rrf_budget_200 | 200.0 (200) | 0.4479 | 0.7296 | 0.7296 | 0.7296 | 1.0000 | 0.4345 | 0.4345 |
| equal_rrf_budget_400 | 368.7 (393) | 0.4479 | 0.7425 | 0.7425 | 0.7425 | 1.0000 | 0.4585 | 0.4585 |

| regime | setting | base pool | candidates mean (p95) | pool_ceiling@5 | pool_ceiling@20 | fraction of attainable@5 | ANY | FullCov ceiling@20 | missing golds recovered | recovered per added candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| STRUCT | h1_c25 | rrf@50 | 106.0 (145) | 0.9570 | 0.9570 | 0.9570 | 0.9995 | 0.8950 | 1334 | 0.0119 |
| STRUCT | h1_c25 | rrf@200 | 254.0 (292) | 0.9674 | 0.9674 | 0.9674 | 1.0000 | 0.9140 | 1256 | 0.0116 |
| STRUCT | h1_c100 | rrf@50 | 114.3 (182) | 0.9575 | 0.9575 | 0.9575 | 0.9995 | 0.8960 | 1336 | 0.0104 |
| STRUCT | h1_c100 | rrf@200 | 261.7 (327) | 0.9676 | 0.9676 | 0.9676 | 1.0000 | 0.9145 | 1257 | 0.0102 |
| STRUCT | h2_c25 | rrf@50 | 1020.0 (1633) | 0.9629 | 0.9629 | 0.9629 | 0.9995 | 0.9080 | 1362 | 0.0007 |
| STRUCT | h2_c25 | rrf@200 | 1164.4 (1778) | 0.9720 | 0.9720 | 0.9720 | 1.0000 | 0.9245 | 1279 | 0.0007 |
| NER | h1_c25 | rrf@50 | 153.2 (216) | 0.7809 | 0.7809 | 0.7809 | 0.9995 | 0.5165 | 412 | 0.0020 |
| NER | h1_c25 | rrf@200 | 300.8 (364) | 0.8004 | 0.8004 | 0.8004 | 1.0000 | 0.5520 | 376 | 0.0019 |
| NER | h1_c100 | rrf@50 | 208.1 (346) | 0.7916 | 0.7916 | 0.7916 | 0.9995 | 0.5335 | 472 | 0.0015 |
| NER | h1_c100 | rrf@200 | 355.2 (491) | 0.8109 | 0.8109 | 0.8109 | 1.0000 | 0.5685 | 435 | 0.0014 |
| NER | h2_c25 | rrf@50 | 1372.8 (2040) | 0.8086 | 0.8086 | 0.8086 | 0.9995 | 0.5680 | 551 | 0.0002 |
| NER | h2_c25 | rrf@200 | 1516.9 (2187) | 0.8265 | 0.8265 | 0.8265 | 1.0000 | 0.6005 | 507 | 0.0002 |
| KNN | h1_c25 | rrf@50 | 81.3 (101) | 0.7252 | 0.7252 | 0.7252 | 0.9995 | 0.4240 | 113 | 0.0018 |
| KNN | h1_c25 | rrf@200 | 227.1 (245) | 0.7458 | 0.7458 | 0.7458 | 1.0000 | 0.4610 | 83 | 0.0015 |
| KNN | h1_c100 | rrf@50 | 81.4 (101) | 0.7252 | 0.7252 | 0.7252 | 0.9995 | 0.4240 | 113 | 0.0018 |
| KNN | h1_c100 | rrf@200 | 227.2 (245) | 0.7458 | 0.7458 | 0.7458 | 1.0000 | 0.4610 | 83 | 0.0015 |
| KNN | h2_c25 | rrf@50 | 212.5 (305) | 0.7476 | 0.7476 | 0.7476 | 1.0000 | 0.4660 | 218 | 0.0007 |
| KNN | h2_c25 | rrf@200 | 353.3 (445) | 0.7655 | 0.7655 | 0.7655 | 1.0000 | 0.4985 | 177 | 0.0006 |
| FULL | h1_c25 | rrf@50 | 221.3 (309) | 0.9653 | 0.9653 | 0.9653 | 0.9995 | 0.9135 | 1374 | 0.0040 |
| FULL | h1_c25 | rrf@200 | 364.7 (452) | 0.9742 | 0.9742 | 0.9742 | 1.0000 | 0.9300 | 1290 | 0.0039 |
| FULL | h1_c100 | rrf@50 | 280.8 (442) | 0.9663 | 0.9663 | 0.9663 | 0.9995 | 0.9150 | 1379 | 0.0030 |
| FULL | h1_c100 | rrf@200 | 423.5 (585) | 0.9752 | 0.9752 | 0.9752 | 1.0000 | 0.9315 | 1295 | 0.0029 |
| FULL | h2_c25 | rrf@50 | 1983.4 (2040) | 0.9742 | 0.9742 | 0.9742 | 1.0000 | 0.9320 | 1414 | 0.0004 |
| FULL | h2_c25 | rrf@200 | 2118.9 (2186) | 0.9820 | 0.9820 | 0.9820 | 1.0000 | 0.9465 | 1325 | 0.0003 |

## Served graph families as loaded

| dataset | family | edges stored | directed as stored | undirected CSR entries | CSR build s |
|---|---|---|---|---|---|
| metaqa | structural | 133,582 | yes | 249,349 | 0.1 |
| metaqa | ner | 5,949 | no (one row per pair) | 11,898 | 0.0 |
| metaqa | knn | 97,920 | no (one row per pair) | 195,840 | 0.1 |
| squad | structural | 874,190 | yes | 1,489,468 | 1.6 |
| squad | ner | 156,130 | no (one row per pair) | 312,260 | 0.1 |
| squad | knn | 42,969 | no (one row per pair) | 85,938 | 0.0 |
| musique | structural | 2,744,076 | yes | 5,302,560 | 9.8 |
| musique | ner | 800,278 | no (one row per pair) | 1,600,556 | 1.5 |
| musique | knn | 265,366 | no (one row per pair) | 530,732 | 0.5 |
| hotpotqa | structural | 15,367,541 | yes | 30,190,574 | 42.9 |
| hotpotqa | ner | 20,369,074 | no (one row per pair) | 40,738,148 | 81.6 |
| hotpotqa | knn | 11,946,289 | no (one row per pair) | 23,892,578 | 57.9 |
| 2wiki | structural | 28,963,600 | yes | 56,417,714 | 111.8 |
| 2wiki | ner | 34,067,241 | no (one row per pair) | 68,134,482 | 174.3 |
| 2wiki | knn | 13,643,063 | no (one row per pair) | 27,286,126 | 70.8 |
| webqsp | structural | 8,309,195 | yes | 11,816,576 | 14.6 |
| webqsp | ner | 2,994,802 | no (one row per pair) | 5,989,604 | 7.1 |
| webqsp | knn | 6,470,520 | no (one row per pair) | 12,941,040 | 19.5 |

These rows supersede, by citation, the stale figures in `docs/M3A_GRAPH_SUBSTRATE.md` and `outputs/m3a/graph_substrate_stats.json` (musique kNN 266,488 / max degree 1,475; 2wiki with no KNN; nothing stored reciprocally): the served musique kNN has 265,366 rows, 2wiki has a kNN family, and every NER/kNN family is stored one row per unordered pair and symmetrised in memory. The old files are not edited (`amendment_1_2026_09_13.stale_rows_withdrawn`).


## Cost

| dataset | wall s | populations_and_gold s | caches s | graph_csr s | eval/retrieval s | eval/graph_regimes s | legacy_continuity/retrieval s | legacy_continuity/graph_regimes s | reachability s | oracle_exposure s |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 855 | 10.8 | 6.5 | 0.3 | 186.1 | 323.3 | 7.5 | 14.3 | 293.6 | 11.7 |
| squad | 451 | 7.1 | 7.1 | 1.7 | 79.5 | 281.5 | 13.4 | 42.1 | 16.1 | – |
| musique | 219 | 3.5 | 0.9 | 11.7 | 13.4 | 45.2 | 10.6 | 36.9 | 96.2 | – |
| hotpotqa | 449 | 39.6 | 1.7 | 182.3 | 37.7 | 111.5 | 8.8 | 27.6 | 39.6 | – |
| 2wiki | 1165 | 79.8 | 10.7 | 357.0 | 61.8 | 234.8 | 8.8 | 39.7 | 371.9 | – |
| webqsp | 445 | 15.1 | 0.1 | 41.2 | 6.0 | 11.7 | – | – | 287.3 | 83.8 |

GPU seconds: 0.


## What this report does not contain

- No ranker, QLS-U or GNN number: the ceilings bound what any ranker over these pools can reach; they say nothing about which ranker reaches it.
- No test-split number.
- No pool larger than the exposure of the inherited protocol: the widest retrieval pool is the 400-max union of two top-200 lists; the widest graph pool is the declared two-hop expansion capped at 2,000 visited nodes.
- No averaging across datasets, and no webqsp figure on the same footing as a text-corpus figure (corpus ceiling column).
- No selection: the knee is a reading of the declared rule; the candidate contract freezes in the M3B declaration, after review.


**STOP_FOR_REVIEW.**
