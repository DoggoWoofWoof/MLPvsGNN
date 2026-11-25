### Gates and cost

| dataset | G-R (PL2's 804 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q x 804 cells = 180,096 elements, mismatches 0/0) | 3, 46 s, 0.28 GB |
| squad | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 2,032 s, 0.33 GB |
| hotpotqa | not run | NOT READ |
| 2wiki | not run | NOT READ |
| musique | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,054 s, 0.63 GB |
| metaqa | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 1,982 s, 0.39 GB |
| webqsp | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,122 s, 5.64 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 432, squad PASS 432; G-X3 PASS, PASS); code-path sha256 `e9d86095...`.

### Dev reading (the rule's own numbers, dev carve, frozen lists): PL2's chosen cell against the union grid's

| dataset | dev q | PL2 R95 | PL3 R95 (all cells) | PL2 R98 | PL3 R98 | PL2's pick re-read = its filed pick |
|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b60` (0.9866 at 60.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | True |
| hotpotqa | NOT READ | | | | | |
| 2wiki | NOT READ | | | | | |
| musique | 1534 | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | True |
| metaqa | 1497 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | True |

### The chosen cell at 0.95 (R95) and its confirmation on eval: PL2's against the union grid's

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL2 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag PL3 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| squad PL2 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| squad PL3 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| musique PL2 | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | not read |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | not read |
| metaqa PL2 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | not read |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| webqsp PL2 | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | not read |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |

### The chosen cell at 0.98 (R98) and its confirmation on eval: PL2's against the union grid's

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL2 | `rrf1000_b60` | 0.9866 | 60.0 | 0.9681 (60.0) SHORT | 0.9664 (60.0) SHORT |
| multihoprag PL3 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| squad PL2 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| squad PL3 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| musique PL2 | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | not read |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | not read |
| metaqa PL2 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | not read |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| webqsp PL2 | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | not read |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |

### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | not read |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | not read |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | not read |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | not read |

### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL2's 804 cells / the union grid

| dataset | 0.95: PL2 | 0.95: PL3 | 0.98: PL2 | 0.98: PL3 |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b60` 0.9866 at 60.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| musique | none (best 0.8449 at 2,611.7) | none (best 0.9355 at 5,783.4) | none (best 0.8449 at 2,611.7) | none (best 0.9355 at 5,783.4) |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best 0.9613 at 5,003.7) | none (best 0.9653 at 6,268.2) |
| webqsp | none (best 0.8016 at 5,100.2) | none (best 0.8480 at 6,067.3) | none (best 0.8016 at 5,100.2) | none (best 0.8480 at 6,067.3) |

### The highest ALL on dev (the bound is 2,500 mean nodes): PL2's 804 cells / the union grid, within the bound and at any size

| dataset | PL2, within the bound | PL3, within the bound | PL2, any size | PL3, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| musique | `rrf1000_b800+FULL:h1_c100` 0.8012 at 1,082.6 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8735 at 2,488.9 | `rrf1000_b800+FULL:h2_c25` 0.8449 at 2,611.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8958 at 2,157.6 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v2000` 0.7505 at 1,888.2 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+FULL:h3_c100_v5000` 0.8016 at 5,100.2 | `rrf1000_b1600+FULL:h3_c100_v5000~vr` 0.8480 at 6,067.3 |

### What PL3's cells bought (dev frontier of the union grid minus PL2's frontier, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9846 / 0.9846 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9920 / 0.9920 (+0.00) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9953 / 0.9960 (+0.07) | 0.9960 / 0.9967 (+0.07) | 0.9973 / 0.9973 (+0.00) | 0.9973 / 0.9987 (+0.13) | 0.9980 / 0.9993 (+0.13) | 0.9980 / 1.0000 (+0.20) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6271 / 0.6349 (+0.78) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7066 / 0.7373 (+3.06) | 0.7151 / 0.7673 (+5.22) | 0.7881 / 0.7947 (+0.65) | 0.8012 / 0.8246 (+2.35) | 0.8012 / 0.8449 (+4.37) | 0.8012 / 0.8735 (+7.24) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7214 / 0.7308 (+0.94) | 0.7408 / 0.8938 (+15.30) | 0.8958 / 0.8978 (+0.20) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6025 / 0.6058 (+0.33) | 0.6251 / 0.6603 (+3.52) | 0.6251 / 0.6802 (+5.51) | 0.6251 / 0.6802 (+5.51) | 0.6251 / 0.6855 (+6.04) | 0.6676 / 0.7147 (+4.71) | 0.6702 / 0.7339 (+6.37) | 0.7505 / 0.7571 (+0.66) | 0.7505 / 0.7731 (+2.26) |

(each entry: PL2's frontier / the union grid's frontier (points bought); `-` = no cell is that small)

### What each family bought: R95 of PL2's cells plus that family's cells (dev), and the best ALL inside the bound among that family's cells alone

| dataset | PL2 R95 | deeper base | similarity caps | rank cap | both | FS regime | seeds 10/20 | bridge |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 123.2 | = PL2; alone best 1.0000 at 111.5 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 200.0 |
| squad | 0.9519 at 15.0 | = PL2; alone best 0.9987 at 1,928.5 | = PL2; alone best 1.0000 at 2,190.4 | = PL2; alone best 0.9987 at 1,928.5 | = PL2; alone best 1.0000 at 2,190.4 | = PL2; alone best 0.9993 at 1,744.6 | = PL2; alone best 0.9993 at 1,944.7 | = PL2; alone best 0.9973 at 795.7 |
| musique | 0.8012 at 1,082.6 SHORT | 0.8605 at 2,386.3 SHORT; alone best 0.8605 at 2,386.3 | 0.8611 at 2,394.6 SHORT; alone best 0.8611 at 2,394.6 | 0.8527 at 2,400.0 SHORT; alone best 0.8527 at 2,400.0 | 0.8735 at 2,488.9 SHORT; alone best 0.8735 at 2,488.9 | 0.8546 at 2,419.9 SHORT; alone best 0.8546 at 2,419.9 | 0.8566 at 2,144.6 SHORT; alone best 0.8566 at 2,144.6 | = PL2; alone best 0.7608 at 677.1 |
| metaqa | 0.8958 at 2,157.6 SHORT | 0.8978 at 2,339.7 SHORT; alone best 0.8978 at 2,339.7 | = PL2; alone best 0.7522 at 2,410.0 | = PL2; alone best 0.7548 at 2,465.9 | = PL2; alone best 0.7522 at 2,410.0 | = PL2; alone best 0.8904 at 2,335.5 | = PL2; alone best 0.7428 at 2,286.6 | = PL2; alone best 0.3621 at 452.7 |
| webqsp | 0.7505 at 1,888.2 SHORT | 0.7532 at 2,246.5 SHORT; alone best 0.7532 at 2,246.5 | 0.7624 at 2,289.1 SHORT; alone best 0.7624 at 2,289.1 | 0.7545 at 2,246.5 SHORT; alone best 0.7545 at 2,246.5 | 0.7624 at 2,288.5 SHORT; alone best 0.7624 at 2,288.5 | 0.7724 at 2,367.7 SHORT; alone best 0.7724 at 2,367.7 | 0.7731 at 2,442.4 SHORT; alone best 0.7731 at 2,442.4 | = PL2; alone best 0.6317 at 479.9 |

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

| dataset | status | NEW (PL3 R95) | NEW ALL (nodes) | OLD (PL2 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | NOT READ | | | | | | |
| 2wiki | NOT READ | | | | | | |
| musique | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) | `rrf1000_b800+FULL:h1_c100` | 0.8283 (1,064.2) | +5.05 [+4.18, +5.92] | 2.307 [2.295, 2.320] |
| metaqa | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` | 0.8952 (2,153.5) | +0.15 [+0.07, +0.24] | 1.085 [1.085, 1.085] |
| webqsp | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | `dense_top200+FULL:h2_c100_v2000` | 0.7638 (1,890.8) | +1.66 [-0.33, +3.53] | 1.305 [1.276, 1.333] |

Beside the call (decide nothing): R98 and the `new`-only pick against PL2's R95.

| dataset | R98 status | R98 NEW ALL (nodes) vs PL2 | new-only status | new-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | ABOVE | 0.9788 (58.0) vs 0.9681 (60.0) | LEVEL | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-vr+top50` | 0.9773 (72.8) |
| musique | LEVEL | 0.8788 (2,455.1) vs 0.8283 (1,064.2) | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8952 (2,153.5) | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) |
| webqsp | LEVEL | 0.7804 (2,468.1) vs 0.7638 (1,890.8) | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) |

### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 300 | 36 (0.120) | 38 | 1.06 | 0 (q 0) | 38 (q 36) | 0 (q 0) | 0 | 2,487.2 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 300 | 63 (0.210) | 245 | 3.89 | 0 (q 0) | 245 (q 63) | 0 (q 0) | 0 | 2,421.1 |

