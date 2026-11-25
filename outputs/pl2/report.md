### Dev reading (the rule's own numbers, dev carve, frozen lists)

| dataset | dev q | R95 retrieval-only | R95 with links | R98 retrieval-only | R98 with links | OLD (own construction) |
|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b50` (0.9732 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b60` (0.9866 at 60.0) | `rrf1000_b50+FULL:h1_c100+top25` (0.9955 at 75.0) | `equal_rrf_budget_50` (0.9643 at 50.0) |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `dense_top25+KNN:h1_c100+top25` (0.9579 at 41.4) | `rrf1000_b75` (0.9833 at 75.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `equal_rrf_budget_50` (0.9793 at 50.0) |
| hotpotqa | NOT READ | | | | | |
| 2wiki | NOT READ | | | | | |
| musique | 1534 | `rrf1000_b2000` (0.7353 at 1,698.3, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b2000` (0.7353 at 1,698.3, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `equal_rrf_budget_200+FULL:h2_c25` (0.7679 at 2,093.5) |
| metaqa | 1497 | `rrf1000_b2000` (0.2712 at 1,894.9, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b2000` (0.2712 at 1,894.9, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` (0.8938 at 2,021.7) |
| webqsp | 1507 | `rrf1000_b2000` (0.4154 at 1,666.3, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `rrf1000_b2000` (0.4154 at 1,666.3, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h3_c25_v2000` (0.6967 at 2,118.8) |

### The chosen cell (R95 over all cells) and its confirmation on eval

| dataset | chosen cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v | OLD eval ALL frozen | OLD eval ALL e3v |
|---|---|---|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) | 0.9540 | 0.9451 |
| squad | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read | 0.9798 | not read |
| musique | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | not read | 0.8026 | not read |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | not read | 0.8949 | not read |
| webqsp | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | not read | 0.6880 | not read |

### The same at 0.98 (R98 over all cells) on eval

| dataset | chosen cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag | `rrf1000_b60` | 0.9866 | 60.0 | 0.9681 (60.0) SHORT | 0.9664 (60.0) SHORT |
| squad | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| musique | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | not read |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | not read |
| webqsp | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | not read |

### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here), retrieval-only and with links

| dataset | retrieval-only 0.95 | with links 0.95 | retrieval-only 0.98 | with links 0.98 |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50` 0.9732 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b60` 0.9866 at 60.0 | `rrf1000_b50+FULL:h1_c100+top25` 0.9955 at 75.0 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `dense_top25+KNN:h1_c100+top25` 0.9579 at 41.4 | `rrf1000_b75` 0.9833 at 75.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| musique | none on the grid (best 0.7353) | none on the grid (best 0.8449) | none on the grid (best 0.7353) | none on the grid (best 0.8449) |
| metaqa | none on the grid (best 0.2712) | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | none on the grid (best 0.2712) | none on the grid (best 0.9613) |
| webqsp | none on the grid (best 0.4154) | none on the grid (best 0.8016) | none on the grid (best 0.4154) | none on the grid (best 0.8016) |

### The highest ALL on dev (any cell; the bound is 2,500 mean nodes), retrieval-only and with links

| dataset | retrieval-only, within the bound | retrieval-only, any size | with links, within the bound | with links, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b125` 1.0000 at 125.0 | `rrf1000_b125` 1.0000 at 125.0 | `rrf1000_b50+FULL:h2_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+FULL:h2_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b800+FULL:h2_c25` 0.9973 at 2,319.9 | `rrf1000_b800+FULL:h2_c25` 0.9973 at 2,319.9 |
| musique | `rrf1000_b2000` 0.7353 at 1,698.3 | `rrf1000_b2000` 0.7353 at 1,698.3 | `rrf1000_b800+FULL:h1_c100` 0.8012 at 1,082.6 | `rrf1000_b800+FULL:h2_c25` 0.8449 at 2,611.7 |
| metaqa | `rrf1000_b2000` 0.2712 at 1,894.9 | `rrf1000_b2000` 0.2712 at 1,894.9 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8958 at 2,157.6 | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 |
| webqsp | `rrf1000_b2000` 0.4154 at 1,666.3 | `rrf1000_b2000` 0.4154 at 1,666.3 | `dense_top200+FULL:h2_c100_v2000` 0.7505 at 1,888.2 | `dense_top200+FULL:h3_c100_v5000` 0.8016 at 5,100.2 |

### What the links bought (dev frontier of the graph cells minus the retrieval cells, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 500 | 1000 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / none | 0.9732 / 0.9777 (+0.45) | 0.9866 / 0.9955 (+0.89) | 0.9955 / 1.0000 (+0.45) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / none | 0.9793 / 0.9773 (-0.20) | 0.9833 / 0.9846 (+0.13) | 0.9887 / 0.9853 (-0.33) | 0.9920 / 0.9907 (-0.13) | 0.9933 / 0.9913 (-0.20) | 0.9940 / 0.9947 (+0.07) | 0.9953 / 0.9953 (+0.00) | 0.9973 / 0.9967 (-0.07) | 0.9980 / 0.9967 (-0.13) | 0.9980 / 0.9973 (-0.07) |
| musique | 0.2979 / none | 0.3696 / 0.4146 (+4.50) | 0.4153 / 0.4733 (+5.80) | 0.4407 / 0.5261 (+8.54) | 0.4778 / 0.5906 (+11.28) | 0.5111 / 0.6271 (+11.60) | 0.5600 / 0.6734 (+11.34) | 0.6206 / 0.7066 (+8.60) | 0.6812 / 0.7881 (+10.69) | 0.7353 / 0.8012 (+6.58) | 0.7353 / 0.8012 (+6.58) |
| metaqa | 0.0147 / none | 0.0200 / 0.2325 (+21.24) | 0.0234 / 0.3039 (+28.06) | 0.0261 / 0.3267 (+30.06) | 0.0321 / 0.3280 (+29.59) | 0.0588 / 0.3300 (+27.12) | 0.0675 / 0.3641 (+29.66) | 0.1102 / 0.5638 (+45.36) | 0.1864 / 0.6927 (+50.63) | 0.2712 / 0.7408 (+46.96) | 0.2712 / 0.8958 (+62.46) |
| webqsp | 0.1526 / none | 0.1772 / 0.3676 (+19.04) | 0.1977 / 0.4625 (+26.48) | 0.2150 / 0.5601 (+34.51) | 0.2455 / 0.5886 (+34.31) | 0.2694 / 0.6025 (+33.31) | 0.3026 / 0.6251 (+32.25) | 0.3278 / 0.6251 (+29.73) | 0.3988 / 0.6676 (+26.87) | 0.4154 / 0.7505 (+33.51) | 0.4154 / 0.7505 (+33.51) |

(each entry: retrieval-only frontier / graph frontier (links bought in points); `none` = no graph cell is that small)

### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

### The call: INCOMPLETE (read 5 of 7: metaqa, multihoprag, musique, squad, webqsp)

| dataset | status | NEW | NEW ALL (nodes) | OLD | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | ABOVE | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `equal_rrf_budget_50` | 0.9540 (50.0) | +1.50 [+0.53, +2.48] | 1.000 [1.000, 1.000] |
| squad | ABOVE | `rrf1000_b15` | 0.9554 (15.0) | `equal_rrf_budget_50` | 0.9798 (50.0) | -2.44 [-2.71, -2.20] | 0.300 [0.300, 0.300] |
| hotpotqa | NOT READ | | | | | | |
| 2wiki | NOT READ | | | | | | |
| musique | ABOVE | `rrf1000_b800+FULL:h1_c100` | 0.8283 (1,064.2) | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | +2.57 [+1.41, +3.72] | 0.509 [0.506, 0.511] |
| metaqa | LEVEL | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` | 0.8952 (2,153.5) | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | +0.03 [+0.00, +0.07] | 1.067 [1.067, 1.067] |
| webqsp | ABOVE | `dense_top200+FULL:h2_c100_v2000` | 0.7638 (1,890.8) | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | +7.58 [+5.92, +9.38] | 0.892 [0.883, 0.902] |
