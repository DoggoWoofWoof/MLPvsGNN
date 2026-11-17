# AP4: the label precision ledger (2026-11-16T21:57:33Z)

Declared in `docs/AP4_LABEL_PRECISION_LEDGER.md`. Graph level; no model is read. Precision, reach and rank1 are in %; lift is precision over the base rate (golds outside rrf's top five per row outside it). Intervals: 95%, 2,000 resamples of questions, seed 0.

## Expectations (section 3; they decide nothing)

| id | expectation | call | per read |
|---|---|---|---|
| E1 | On 2Wiki and HotpotQA, R.mention has the highest top1 precision of D83's labels. | NOT_HELD | [False, False, True, False] |
| E2 | K.knn has the lowest top1 lift of D83's labels on every passage dataset. | NOT_HELD | [False, False, False, False, False, False, False] |
| E3 | Pairs with two or more labels are ABOVE single-label pairs in top1 precision, on every dataset. | NOT_HELD | [True, True, True, True, False, True, True, True, True] |
| E4 | On MetaQA and WebQSP, S has the highest top1 precision of D83's labels. | HELD | [True, True] |
| E5 | MuSiQue's found-anchored reach of any link falls with the hops (2 > 3 > 4). | HELD | [True, True] |
| E6 | D83's R is not BELOW the record's R in top1 precision on any paired dataset. | NOT_HELD | [True, False, False, True, True, True, True, False, False] |

## B1V-B1W: D83 minus the record, same questions and pools

| dataset | group | top1 precision | top1 price | found reach | found rank1 |
|---|---|---|---|---|---|
| musique | group:R | 0.0 [-0.1, 0.1] LEVEL | 2.10 [1.48, 2.72] | 3.1 [1.3, 5.1] ABOVE | -4.7 [-7.2, -2.3] BELOW |
| musique | group:K | 0.0 [0.0, 0.0] LEVEL | -0.00 [-0.01, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| musique | any | -0.0 [-0.1, 0.1] LEVEL | 1.90 [1.30, 2.52] | 2.2 [0.6, 3.9] ABOVE | -1.5 [-3.1, 0.2] LEVEL |
| musique | K pairs at the anchors | {'d83': 9546, 'record': 9562, 'both': 9546, 'same': False} | | | |
| hotpotqa | group:R | -0.7 [-0.9, -0.4] BELOW | 0.88 [0.70, 1.06] | 5.7 [3.0, 8.7] ABOVE | -4.9 [-9.3, -0.5] BELOW |
| hotpotqa | group:K | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| hotpotqa | any | -0.4 [-0.5, -0.2] BELOW | 0.80 [0.63, 0.97] | 3.7 [1.6, 6.2] ABOVE | -0.6 [-3.5, 2.2] LEVEL |
| hotpotqa | K pairs at the anchors | {'d83': 6240, 'record': 6240, 'both': 6240, 'same': True} | | | |
| 2wiki | group:R | -0.7 [-1.3, -0.1] BELOW | 0.20 [0.04, 0.36] | 0.4 [-0.1, 1.1] LEVEL | -3.8 [-6.5, -1.0] BELOW |
| 2wiki | group:K | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| 2wiki | any | -0.3 [-0.6, -0.0] BELOW | 0.17 [0.02, 0.32] | 0.0 [0.0, 0.0] LEVEL | -1.8 [-3.4, -0.1] BELOW |
| 2wiki | K pairs at the anchors | {'d83': 6093, 'record': 6093, 'both': 6093, 'same': True} | | | |

## D83E-REC: D83 minus the record, same questions and pools

| dataset | group | top1 precision | top1 price | found reach | found rank1 |
|---|---|---|---|---|---|
| squad | group:R | 0.0 [-0.0, 0.0] LEVEL | 0.12 [0.10, 0.14] | - None | - None |
| squad | group:K | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | - None | - None |
| squad | any | 0.0 [-0.0, 0.0] LEVEL | 0.12 [0.10, 0.14] | - None | - None |
| squad | K pairs at the anchors | {'d83': 43356, 'record': 43356, 'both': 43356, 'same': True} | | | |
| hotpotqa | group:R | 0.1 [0.1, 0.1] ABOVE | -1.01 [-1.18, -0.85] | 1.0 [0.5, 1.5] ABOVE | -1.8 [-2.7, -0.8] BELOW |
| hotpotqa | group:K | 0.0 [0.0, 0.0] LEVEL | -0.00 [-0.00, -0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| hotpotqa | any | 0.1 [0.1, 0.1] ABOVE | -1.05 [-1.22, -0.89] | 0.7 [0.2, 1.2] ABOVE | -1.0 [-1.7, -0.4] BELOW |
| hotpotqa | K pairs at the anchors | {'d83': 79918, 'record': 79938, 'both': 79918, 'same': False} | | | |
| 2wiki | group:R | 0.1 [0.1, 0.1] ABOVE | -1.57 [-1.74, -1.39] | 0.8 [0.6, 1.0] ABOVE | -1.5 [-1.9, -1.2] BELOW |
| 2wiki | group:K | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| 2wiki | any | 0.1 [0.1, 0.1] ABOVE | -1.65 [-1.82, -1.47] | 0.6 [0.4, 0.8] ABOVE | -0.5 [-0.8, -0.3] BELOW |
| 2wiki | K pairs at the anchors | {'d83': 122403, 'record': 122410, 'both': 122403, 'same': False} | | | |
| musique | group:R | 0.1 [0.0, 0.1] ABOVE | 0.27 [-0.10, 0.61] | 2.6 [1.6, 3.5] ABOVE | -4.1 [-5.8, -2.4] BELOW |
| musique | group:K | 0.1 [0.0, 0.1] ABOVE | -0.08 [-0.10, -0.06] | 0.0 [0.0, 0.0] LEVEL | 0.4 [0.0, 1.3] LEVEL |
| musique | any | 0.0 [0.0, 0.1] ABOVE | 0.08 [-0.28, 0.42] | 2.2 [1.2, 3.1] ABOVE | -1.3 [-2.5, -0.1] BELOW |
| musique | K pairs at the anchors | {'d83': 21292, 'record': 21631, 'both': 21292, 'same': False} | | | |
| metaqa | group:S | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | - None |
| metaqa | group:R | -0.3 [-0.6, -0.0] BELOW | 0.44 [0.39, 0.49] | 0.0 [-0.4, 0.3] LEVEL | -41.7 [-100.0, 33.3] LEVEL |
| metaqa | group:K | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| metaqa | any | -0.2 [-0.3, -0.2] BELOW | 0.38 [0.33, 0.42] | -0.1 [-0.4, 0.0] LEVEL | -1.8 [-8.9, 0.0] LEVEL |
| metaqa | K pairs at the anchors | {'d83': 31431, 'record': 31431, 'both': 31431, 'same': True} | | | |
| webqsp | group:S | 0.0 [0.0, 0.0] LEVEL | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| webqsp | group:R | -1.3 [-1.9, -0.7] BELOW | -1.00 [-1.36, -0.65] | 0.3 [0.0, 0.9] LEVEL | - None |
| webqsp | group:K | 0.0 [0.0, 0.0] LEVEL | -0.00 [-0.00, 0.00] | 0.0 [0.0, 0.0] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| webqsp | any | 0.0 [0.0, 0.0] ABOVE | -0.26 [-0.53, -0.00] | 0.1 [0.0, 0.3] LEVEL | 0.0 [0.0, 0.0] LEVEL |
| webqsp | K pairs at the anchors | {'d83': 9120, 'record': 9122, 'both': 9120, 'same': False} | | | |

## B1V musique (1000 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 8.2829 | 2.2 [1.9, 2.6] | 67.01 [56.43, 78.85] | POINTS | 0.4 [0.1, 0.7] ABOVE | 23.3 [20.7, 26.0] | 59.6 [53.6, 65.9] | 0.72549 | 0.62353 |
| label:R.mention | top1 | 10.16 | 2.2 [1.7, 2.8] | 61.36 [47.68, 78.70] | POINTS | 0.2 [-0.1, 0.6] LEVEL | - | 58.3 [51.7, 64.8] | 0.69444 | 0.59259 |
| label:R.name_comention | found | 18.5827 | 2.2 [2.0, 2.4] | 67.35 [62.17, 72.82] | POINTS | 0.4 [0.3, 0.5] ABOVE | 49.5 [46.5, 52.6] | 32.4 [28.5, 36.6] | 0.48921 | 0.36511 |
| label:R.name_comention | top1 | 18.919 | 2.4 [2.2, 2.7] | 67.66 [62.03, 74.13] | POINTS | 0.4 [0.3, 0.6] ABOVE | - | 32.9 [28.6, 37.6] | 0.50469 | 0.38028 |
| label:R.title_comention | found | 1.2645 | 3.0 [2.2, 3.8] | 89.83 [67.30, 114.61] | POINTS | 1.1 [0.4, 2.0] ABOVE | 4.9 [3.6, 6.2] | 73.6 [61.1, 84.2] | 0.83019 | 0.79245 |
| label:R.title_comention | top1 | 1.175 | 4.1 [2.9, 5.4] | 113.69 [81.20, 148.34] | POINTS | 2.1 [1.0, 3.3] ABOVE | - | 83.0 [71.2, 92.9] | 0.93617 | 0.82979 |
| label:K.knn | found | 3.9802 | 3.5 [3.0, 3.9] | 105.00 [92.15, 119.10] | POINTS | 1.6 [1.2, 2.1] ABOVE | 18.3 [16.0, 20.7] | 36.4 [29.7, 43.1] | 0.69744 | 0.41538 |
| label:K.knn | top1 | 3.918 | 4.6 [4.0, 5.2] | 127.14 [110.30, 144.63] | POINTS | 2.6 [2.0, 3.2] ABOVE | - | 40.7 [33.1, 47.8] | 0.72316 | 0.42938 |
| group:R | found | 25.2419 | 2.0 [1.8, 2.1] | 59.26 [54.60, 64.39] | POINTS | 0.1 [0.1, 0.2] ABOVE | 58.7 [55.7, 61.7] | 30.3 [26.7, 34.1] | 0.44729 | 0.34789 |
| group:R | top1 | 27.133 | 2.1 [1.8, 2.3] | 57.64 [51.46, 64.84] | POINTS | 0.1 [0.0, 0.1] ABOVE | - | 30.9 [27.2, 35.0] | 0.46109 | 0.3463 |
| group:K | found | 3.9802 | 3.5 [3.0, 3.9] | 105.00 [92.15, 119.10] | POINTS | 1.6 [1.2, 2.1] ABOVE | 18.3 [16.0, 20.7] | 36.4 [29.7, 43.1] | 0.69744 | 0.41538 |
| group:K | top1 | 3.918 | 4.6 [4.0, 5.2] | 127.14 [110.30, 144.63] | POINTS | 2.6 [2.0, 3.2] ABOVE | - | 40.7 [33.1, 47.8] | 0.72316 | 0.42938 |
| any | found | 27.5113 | 1.8 [1.7, 2.0] | 55.46 [51.24, 60.07] | POINTS | - | 59.8 [56.8, 62.9] | 19.7 [16.6, 23.0] | 0.33876 | 0.23669 |
| any | top1 | 29.303 | 2.0 [1.8, 2.2] | 55.75 [49.99, 62.08] | POINTS | - | - | 20.4 [17.0, 24.0] | 0.36142 | 0.23034 |

Multi-label minus single-label precision (found): 3.7 [3.1, 4.4] ABOVE (5743 / 33158 pairs).

Multi-label minus single-label precision (top1): 4.4 [3.7, 5.2] ABOVE (4304 / 24999 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.name_comention | 56.1 | 1.6 [1.4, 1.7] | 47.04 [42.46, 51.81] |
| found | R.mention | 19.0 | 0.7 [0.5, 0.9] | 20.87 [14.64, 28.31] |
| found | K.knn | 8.2 | 0.4 [0.2, 0.7] | 13.22 [6.71, 21.11] |
| found | R.mention + R.name_comention | 6.2 | 3.8 [3.0, 4.7] | 115.24 [90.92, 142.35] |
| found | R.name_comention + K.knn | 3.6 | 5.4 [4.2, 6.6] | 163.39 [128.21, 200.51] |
| found | R.mention + R.title_comention | 2.4 | 1.6 [0.9, 2.6] | 48.82 [25.97, 78.37] |
| found | R.title_comention | 1.8 | 3.0 [1.8, 4.3] | 90.14 [55.93, 130.51] |
| found | R.mention + R.name_comention + K.knn | 1.7 | 12.1 [9.6, 14.9] | 365.50 [290.32, 448.08] |
| top1 | R.name_comention | 53.0 | 1.6 [1.4, 1.8] | 44.82 [39.94, 50.14] |
| top1 | R.mention | 23.5 | 0.7 [0.4, 1.0] | 18.22 [11.85, 27.18] |
| top1 | K.knn | 7.4 | 1.2 [0.7, 1.6] | 32.06 [20.01, 45.21] |
| top1 | R.mention + R.name_comention | 6.5 | 4.2 [3.3, 5.4] | 117.59 [91.58, 149.80] |
| top1 | R.name_comention + K.knn | 3.4 | 6.8 [5.3, 8.4] | 189.24 [149.65, 236.27] |
| top1 | R.mention + R.title_comention | 2.2 | 2.2 [1.1, 3.5] | 60.97 [29.55, 98.68] |
| top1 | R.mention + R.name_comention + K.knn | 1.6 | 12.7 [9.7, 15.9] | 354.40 [269.61, 448.41] |
| top1 | R.title_comention | 1.5 | 4.2 [2.4, 6.4] | 116.22 [66.60, 176.89] |

| hops | kind | unit | precision | lift | reach | rank1 |
|---|---|---|---|---|---|---|
| 2 | found | any | 1.6 [1.4, 1.8] | 88.97 [80.80, 98.15] | 84.4 [80.1, 88.3] | 27.0 [21.5, 32.6] |
| 2 | found | group:R | 1.7 [1.5, 2.0] | 96.58 [87.06, 107.60] | 82.3 [77.8, 86.4] | 43.4 [37.2, 49.8] |
| 2 | found | group:K | 3.4 [2.7, 4.1] | 190.12 [155.46, 224.77] | 30.3 [24.9, 35.5] | 43.8 [33.8, 54.0] |
| 2 | top1 | any | 2.0 [1.8, 2.4] | 101.12 [89.31, 114.90] | - | 26.2 [20.7, 31.9] |
| 2 | top1 | group:R | 2.2 [1.9, 2.6] | 108.01 [94.25, 124.78] | - | 40.9 [34.4, 47.2] |
| 2 | top1 | group:K | 4.8 [3.9, 5.7] | 235.90 [193.20, 279.55] | - | 41.5 [31.7, 51.2] |
| 3 | found | any | 1.8 [1.5, 2.0] | 46.63 [40.97, 53.17] | 57.7 [53.5, 61.8] | 15.2 [10.6, 20.0] |
| 3 | found | group:R | 1.8 [1.6, 2.1] | 49.04 [42.86, 56.05] | 56.7 [52.4, 60.8] | 21.5 [16.3, 27.0] |
| 3 | found | group:K | 3.5 [2.8, 4.2] | 92.70 [75.07, 110.70] | 18.4 [14.9, 21.9] | 27.1 [16.7, 38.2] |
| 3 | top1 | any | 1.7 [1.4, 2.1] | 42.40 [34.13, 51.97] | - | 16.2 [11.0, 21.7] |
| 3 | top1 | group:R | 1.8 [1.4, 2.2] | 43.13 [34.22, 53.38] | - | 21.2 [15.4, 27.2] |
| 3 | top1 | group:K | 4.4 [3.4, 5.6] | 108.05 [84.02, 133.08] | - | 36.4 [23.9, 49.4] |
| 4 | found | any | 2.5 [2.2, 2.9] | 35.33 [30.42, 40.74] | 43.1 [38.1, 48.4] | 15.8 [9.9, 22.3] |
| 4 | found | group:R | 2.7 [2.3, 3.1] | 37.68 [32.26, 43.70] | 42.6 [37.6, 47.7] | 24.8 [17.9, 32.0] |
| 4 | found | group:K | 3.7 [2.6, 4.8] | 51.29 [36.22, 67.80] | 9.1 [6.5, 12.1] | 36.1 [21.2, 51.6] |
| 4 | top1 | any | 2.4 [2.1, 2.8] | 32.51 [27.74, 38.10] | - | 14.7 [8.3, 20.8] |
| 4 | top1 | group:R | 2.4 [2.0, 2.9] | 32.60 [27.51, 38.65] | - | 25.2 [16.8, 33.6] |
| 4 | top1 | group:K | 4.2 [2.8, 5.7] | 56.48 [38.87, 75.63] | - | 46.4 [27.3, 65.0] |

## B1V hotpotqa (1000 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 0.7743 | 12.0 [10.0, 14.1] | 29.07 [25.51, 32.78] | POINTS | 9.7 [7.9, 11.6] ABOVE | 64.5 [58.3, 70.3] | 89.2 [84.3, 93.9] | 0.94937 | 0.92405 |
| label:R.mention | top1 | 0.759 | 19.0 [16.0, 22.5] | 42.49 [36.78, 49.75] | POINTS | 15.4 [12.7, 18.6] ABOVE | - | 88.8 [83.4, 93.7] | 0.95105 | 0.93007 |
| label:R.name_comention | found | 4.2034 | 2.3 [1.9, 2.6] | 5.46 [4.86, 6.08] | POINTS | -0.0 [-0.2, 0.2] LEVEL | 65.7 [59.5, 71.7] | 49.7 [42.1, 57.3] | 0.67702 | 0.5528 |
| label:R.name_comention | top1 | 4.463 | 3.6 [3.0, 4.1] | 8.03 [7.11, 8.98] | POINTS | 0.0 [-0.3, 0.3] LEVEL | - | 50.0 [42.6, 57.8] | 0.6859 | 0.55128 |
| label:R.title_comention | found | 0.1728 | 13.6 [9.4, 18.6] | 32.96 [23.38, 44.42] | POINTS | 11.3 [7.3, 16.2] ABOVE | 16.3 [11.8, 20.9] | 97.5 [92.0, 100.0] | 0.975 | 0.975 |
| label:R.title_comention | top1 | 0.161 | 19.9 [13.3, 27.7] | 44.52 [29.94, 61.20] | POINTS | 16.3 [9.9, 24.0] ABOVE | - | 96.9 [89.7, 100.0] | 0.96875 | 0.96875 |
| label:K.knn | found | 2.2969 | 3.3 [2.7, 3.9] | 7.94 [6.91, 8.89] | POINTS | 1.0 [0.6, 1.4] ABOVE | 52.2 [45.7, 58.1] | 53.9 [44.8, 62.5] | 0.8125 | 0.53906 |
| label:K.knn | top1 | 2.333 | 5.2 [4.4, 6.1] | 11.71 [10.11, 13.33] | POINTS | 1.7 [1.0, 2.3] ABOVE | - | 53.8 [44.9, 62.6] | 0.78151 | 0.55462 |
| group:R | found | 4.4268 | 2.8 [2.4, 3.3] | 6.89 [6.33, 7.46] | POINTS | 0.6 [0.5, 0.7] ABOVE | 87.3 [83.1, 91.4] | 48.1 [41.4, 54.5] | 0.64019 | 0.53271 |
| group:R | top1 | 4.671 | 4.3 [3.8, 4.9] | 9.73 [8.85, 10.68] | POINTS | 0.8 [0.6, 1.0] ABOVE | - | 49.7 [43.1, 56.6] | 0.65829 | 0.53769 |
| group:K | found | 2.2969 | 3.3 [2.7, 3.9] | 7.94 [6.91, 8.89] | POINTS | 1.0 [0.6, 1.4] ABOVE | 52.2 [45.7, 58.1] | 53.9 [44.8, 62.5] | 0.8125 | 0.53906 |
| group:K | top1 | 2.333 | 5.2 [4.4, 6.1] | 11.71 [10.11, 13.33] | POINTS | 1.7 [1.0, 2.3] ABOVE | - | 53.8 [44.9, 62.6] | 0.78151 | 0.55462 |
| any | found | 5.8301 | 2.3 [2.0, 2.6] | 5.47 [5.11, 5.84] | POINTS | - | 91.4 [87.6, 94.8] | 35.7 [29.5, 41.7] | 0.58036 | 0.40179 |
| any | top1 | 6.064 | 3.6 [3.1, 4.0] | 7.98 [7.38, 8.61] | POINTS | - | - | 38.4 [32.2, 44.8] | 0.6019 | 0.40758 |

Multi-label minus single-label precision (found): 8.0 [6.8, 9.3] ABOVE (2143 / 7774 pairs).

Multi-label minus single-label precision (top1): 11.8 [9.9, 13.7] ABOVE (1293 / 4771 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.name_comention | 53.0 | 0.4 [0.3, 0.6] | 1.06 [0.67, 1.52] |
| found | K.knn | 24.1 | 0.4 [0.2, 0.7] | 1.02 [0.42, 1.67] |
| found | R.name_comention + K.knn | 8.8 | 3.3 [2.2, 4.6] | 8.08 [5.53, 11.07] |
| found | R.mention + R.name_comention + K.knn | 5.3 | 12.5 [9.5, 15.7] | 30.34 [23.79, 37.48] |
| found | R.mention + R.name_comention | 5.0 | 8.7 [6.2, 11.5] | 21.18 [15.30, 27.46] |
| found | R.mention + R.title_comention | 1.3 | 16.8 [10.2, 24.5] | 40.69 [25.04, 59.18] |
| found | R.mention + R.title_comention + K.knn | 0.8 | 17.3 [9.5, 27.6] | 41.88 [22.85, 66.80] |
| found | R.title_comention | 0.7 | 4.5 [0.0, 11.5] | 10.85 [0.00, 27.58] |
| found | R.mention | 0.6 | 8.5 [1.7, 16.1] | 20.53 [4.25, 38.55] |
| top1 | R.name_comention | 54.3 | 0.9 [0.6, 1.3] | 2.11 [1.44, 2.82] |
| top1 | K.knn | 23.0 | 0.9 [0.4, 1.5] | 2.09 [1.04, 3.25] |
| top1 | R.name_comention + K.knn | 9.3 | 4.6 [3.0, 6.3] | 10.36 [6.83, 14.15] |
| top1 | R.mention + R.name_comention + K.knn | 5.2 | 20.4 [15.8, 25.5] | 45.78 [36.48, 57.47] |
| top1 | R.mention + R.name_comention | 4.8 | 13.1 [9.2, 18.0] | 29.45 [21.28, 39.57] |
| top1 | R.mention + R.title_comention | 1.1 | 29.7 [18.2, 44.6] | 66.49 [40.96, 101.40] |
| top1 | R.title_comention | 0.8 | 4.0 [0.0, 11.4] | 8.96 [0.00, 25.62] |
| top1 | R.mention + R.title_comention + K.knn | 0.7 | 26.8 [12.7, 46.4] | 60.09 [28.58, 103.98] |

## B1V 2wiki (1000 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 0.537 | 68.5 [65.1, 72.1] | 81.90 [77.29, 86.83] | POINTS | 60.4 [57.2, 63.7] ABOVE | 88.1 [85.4, 90.5] | 93.2 [91.3, 95.0] | 0.99026 | 0.93344 |
| label:R.mention | top1 | 0.594 | 72.1 [68.5, 76.0] | 85.92 [80.36, 91.86] | POINTS | 63.0 [59.6, 66.8] ABOVE | - | 93.0 [90.6, 95.3] | 0.98598 | 0.92523 |
| label:R.name_comention | found | 3.4528 | 9.6 [8.7, 10.5] | 11.48 [10.64, 12.44] | POINTS | 1.5 [1.1, 1.9] ABOVE | 78.9 [75.9, 81.9] | 50.8 [46.7, 55.2] | 0.72793 | 0.50991 |
| label:R.name_comention | top1 | 3.697 | 10.5 [9.6, 11.6] | 12.58 [11.53, 13.78] | POINTS | 1.5 [1.0, 2.0] ABOVE | - | 50.0 [45.2, 55.0] | 0.71795 | 0.51795 |
| label:R.title_comention | found | 0.1099 | 58.7 [50.0, 68.0] | 70.15 [59.93, 81.35] | POINTS | 50.6 [42.0, 59.8] ABOVE | 15.5 [12.9, 18.2] | 97.2 [93.5, 100.0] | 0.99074 | 0.96296 |
| label:R.title_comention | top1 | 0.115 | 67.8 [59.0, 77.4] | 80.88 [70.37, 92.66] | POINTS | 58.8 [50.2, 68.2] ABOVE | - | 97.4 [93.2, 100.0] | 0.98718 | 0.96154 |
| label:K.knn | found | 2.2091 | 9.8 [9.0, 10.8] | 11.76 [10.87, 12.71] | POINTS | 1.7 [1.0, 2.5] ABOVE | 51.9 [48.0, 55.7] | 49.5 [44.2, 54.5] | 0.73901 | 0.43132 |
| label:K.knn | top1 | 2.395 | 12.1 [11.1, 13.2] | 14.44 [13.27, 15.71] | POINTS | 3.1 [2.2, 4.0] ABOVE | - | 47.9 [42.3, 53.4] | 0.74828 | 0.43448 |
| group:R | found | 3.5818 | 11.3 [10.4, 12.4] | 13.50 [12.61, 14.57] | POINTS | 3.2 [2.8, 3.6] ABOVE | 96.4 [94.8, 97.9] | 50.5 [46.8, 54.4] | 0.72083 | 0.50074 |
| group:R | top1 | 3.841 | 12.5 [11.5, 13.6] | 14.93 [13.81, 16.21] | POINTS | 3.5 [3.0, 4.0] ABOVE | - | 50.7 [46.2, 55.0] | 0.7131 | 0.50728 |
| group:K | found | 2.2091 | 9.8 [9.0, 10.8] | 11.76 [10.87, 12.71] | POINTS | 1.7 [1.0, 2.5] ABOVE | 51.9 [48.0, 55.7] | 49.5 [44.2, 54.5] | 0.73901 | 0.43132 |
| group:K | top1 | 2.395 | 12.1 [11.1, 13.2] | 14.44 [13.27, 15.71] | POINTS | 3.1 [2.2, 4.0] ABOVE | - | 47.9 [42.3, 53.4] | 0.74828 | 0.43448 |
| any | found | 5.0161 | 8.1 [7.5, 8.7] | 9.69 [9.19, 10.25] | POINTS | - | 96.8 [95.3, 98.2] | 31.6 [28.1, 35.1] | 0.52717 | 0.2746 |
| any | top1 | 5.359 | 9.0 [8.4, 9.7] | 10.79 [10.17, 11.50] | POINTS | - | - | 32.2 [28.2, 36.4] | 0.55258 | 0.29278 |

Multi-label minus single-label precision (found): 37.5 [35.2, 40.1] ABOVE (1702 / 6695 pairs).

Multi-label minus single-label precision (top1): 40.7 [38.0, 43.7] ABOVE (1113 / 4246 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.name_comention | 50.4 | 0.6 [0.4, 0.9] | 0.76 [0.48, 1.08] |
| found | K.knn | 28.6 | 0.2 [0.0, 0.3] | 0.20 [0.05, 0.42] |
| found | R.name_comention + K.knn | 9.7 | 3.9 [2.7, 5.3] | 4.69 [3.18, 6.36] |
| found | R.mention + R.name_comention + K.knn | 5.0 | 65.7 [61.3, 70.5] | 78.54 [72.48, 85.06] |
| found | R.mention + R.name_comention | 3.7 | 70.7 [64.8, 76.7] | 84.50 [77.20, 91.99] |
| found | R.mention + R.title_comention | 1.1 | 71.4 [60.0, 82.1] | 85.37 [72.06, 98.44] |
| found | R.mention + R.title_comention + K.knn | 0.5 | 91.1 [81.8, 98.0] | 108.90 [97.40, 120.15] |
| top1 | R.name_comention | 50.3 | 0.7 [0.4, 1.0] | 0.84 [0.49, 1.26] |
| top1 | K.knn | 28.3 | 0.3 [0.1, 0.5] | 0.31 [0.08, 0.65] |
| top1 | R.name_comention + K.knn | 9.9 | 6.0 [4.0, 8.0] | 7.20 [4.74, 9.62] |
| top1 | R.mention + R.name_comention + K.knn | 5.6 | 71.1 [66.2, 76.5] | 84.83 [78.06, 92.53] |
| top1 | R.mention + R.name_comention | 3.3 | 72.6 [65.9, 79.4] | 86.54 [78.26, 95.19] |
| top1 | R.mention + R.title_comention | 1.1 | 77.0 [66.1, 88.0] | 91.88 [78.42, 106.19] |
| top1 | R.mention + R.title_comention + K.knn | 0.6 | 93.5 [82.9, 100.0] | 111.55 [98.65, 122.50] |

## B1W musique (1000 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 8.2829 | 2.2 [1.9, 2.6] | 67.01 [56.43, 78.85] | POINTS | 0.4 [0.1, 0.7] ABOVE | 23.3 [20.7, 26.0] | 59.6 [53.6, 65.9] | 0.72549 | 0.62353 |
| label:slot | top1 | 10.156 | 2.2 [1.7, 2.8] | 61.38 [47.71, 78.72] | POINTS | 0.2 [-0.1, 0.6] LEVEL | - | 58.3 [51.7, 64.8] | 0.69444 | 0.59259 |
| label:NER.spacy | found | 17.2334 | 2.2 [2.1, 2.4] | 67.65 [62.03, 73.86] | POINTS | 0.4 [0.3, 0.5] ABOVE | 47.6 [44.5, 50.6] | 39.2 [35.0, 44.0] | 0.56061 | 0.39773 |
| label:NER.spacy | top1 | 17.063 | 2.5 [2.3, 2.7] | 69.48 [62.92, 77.06] | POINTS | 0.5 [0.3, 0.7] ABOVE | - | 40.0 [35.3, 44.8] | 0.56962 | 0.41519 |
| label:K.knn | found | 3.9894 | 3.5 [3.0, 3.9] | 104.76 [91.86, 118.83] | POINTS | 1.6 [1.2, 2.1] ABOVE | 18.3 [16.0, 20.7] | 36.4 [29.7, 43.1] | 0.69744 | 0.41538 |
| label:K.knn | top1 | 3.921 | 4.6 [4.0, 5.2] | 127.05 [110.29, 144.47] | POINTS | 2.6 [2.0, 3.1] ABOVE | - | 40.7 [33.1, 47.8] | 0.72316 | 0.42938 |
| group:R | found | 23.476 | 1.9 [1.8, 2.1] | 58.33 [53.37, 64.04] | POINTS | 0.1 [0.1, 0.1] ABOVE | 55.6 [52.4, 58.5] | 35.0 [31.1, 39.1] | 0.50647 | 0.36731 |
| group:R | top1 | 25.036 | 2.1 [1.8, 2.3] | 57.47 [50.58, 65.27] | POINTS | 0.1 [-0.0, 0.1] LEVEL | - | 36.0 [31.7, 40.1] | 0.51674 | 0.37657 |
| group:K | found | 3.9894 | 3.5 [3.0, 3.9] | 104.76 [91.86, 118.83] | POINTS | 1.6 [1.2, 2.1] ABOVE | 18.3 [16.0, 20.7] | 36.4 [29.7, 43.1] | 0.69744 | 0.41538 |
| group:K | top1 | 3.921 | 4.6 [4.0, 5.2] | 127.05 [110.29, 144.47] | POINTS | 2.6 [2.0, 3.1] ABOVE | - | 40.7 [33.1, 47.8] | 0.72316 | 0.42938 |
| any | found | 25.93 | 1.8 [1.7, 2.0] | 55.13 [50.72, 60.05] | POINTS | - | 57.7 [54.4, 60.7] | 21.2 [18.1, 24.5] | 0.38258 | 0.23328 |
| any | top1 | 27.4 | 2.0 [1.8, 2.3] | 56.06 [49.87, 63.07] | POINTS | - | - | 21.7 [18.2, 25.4] | 0.40711 | 0.24111 |

Multi-label minus single-label precision (found): 4.4 [3.7, 5.3] ABOVE (4448 / 32217 pairs).

Multi-label minus single-label precision (top1): 5.2 [4.3, 6.2] ABOVE (3286 / 24114 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | NER.spacy | 55.4 | 1.6 [1.4, 1.7] | 47.04 [42.19, 52.22] |
| found | slot | 23.1 | 0.8 [0.6, 1.1] | 25.45 [18.89, 32.99] |
| found | K.knn | 9.5 | 0.8 [0.5, 1.1] | 24.45 [15.62, 34.11] |
| found | slot + NER.spacy | 6.2 | 3.8 [2.9, 4.7] | 114.45 [88.71, 141.25] |
| found | NER.spacy + K.knn | 3.2 | 5.5 [4.2, 6.8] | 165.66 [128.73, 207.31] |
| found | slot + NER.spacy + K.knn | 1.7 | 12.8 [10.2, 15.8] | 388.77 [308.26, 477.45] |
| found | slot + K.knn | 1.0 | 6.4 [4.0, 9.2] | 194.46 [118.87, 277.53] |
| top1 | NER.spacy | 51.3 | 1.7 [1.5, 1.9] | 46.15 [40.77, 52.53] |
| top1 | slot | 28.1 | 0.9 [0.6, 1.2] | 23.86 [16.63, 33.97] |
| top1 | K.knn | 8.6 | 1.5 [1.0, 2.0] | 41.20 [28.48, 56.04] |
| top1 | slot + NER.spacy | 6.3 | 4.3 [3.3, 5.5] | 119.11 [91.84, 152.68] |
| top1 | NER.spacy + K.knn | 3.0 | 7.2 [5.6, 9.1] | 201.42 [155.47, 253.91] |
| top1 | slot + NER.spacy + K.knn | 1.7 | 13.0 [10.0, 16.3] | 361.66 [276.65, 458.71] |

| hops | kind | unit | precision | lift | reach | rank1 |
|---|---|---|---|---|---|---|
| 2 | found | any | 1.7 [1.5, 1.9] | 95.25 [85.59, 105.87] | 81.0 [76.6, 85.2] | 26.5 [21.0, 32.2] |
| 2 | found | group:R | 1.9 [1.6, 2.2] | 105.65 [94.34, 118.72] | 78.9 [74.3, 83.3] | 44.8 [38.5, 51.3] |
| 2 | found | group:K | 3.4 [2.7, 4.1] | 190.05 [155.33, 224.58] | 30.3 [24.9, 35.5] | 43.8 [33.8, 54.0] |
| 2 | top1 | any | 2.2 [1.9, 2.6] | 109.09 [95.53, 124.85] | - | 25.7 [19.7, 31.4] |
| 2 | top1 | group:R | 2.4 [2.1, 2.8] | 118.14 [101.91, 138.10] | - | 44.1 [37.3, 50.7] |
| 2 | top1 | group:K | 4.8 [3.9, 5.7] | 235.78 [193.20, 279.49] | - | 41.5 [31.7, 51.2] |
| 3 | found | any | 1.8 [1.5, 2.0] | 46.93 [41.05, 53.70] | 60.4 [56.2, 64.7] | 19.3 [14.4, 24.3] |
| 3 | found | group:R | 1.8 [1.6, 2.1] | 48.67 [42.18, 56.09] | 57.7 [53.4, 62.3] | 28.1 [22.4, 34.1] |
| 3 | found | group:K | 3.5 [2.8, 4.2] | 92.19 [74.64, 109.90] | 18.4 [14.9, 21.9] | 27.1 [16.7, 38.2] |
| 3 | top1 | any | 1.7 [1.4, 2.1] | 41.52 [33.05, 51.18] | - | 20.2 [14.4, 26.5] |
| 3 | top1 | group:R | 1.7 [1.3, 2.1] | 41.53 [32.42, 51.93] | - | 28.1 [21.4, 35.1] |
| 3 | top1 | group:K | 4.4 [3.4, 5.5] | 107.88 [83.89, 133.03] | - | 36.4 [23.9, 49.4] |
| 4 | found | any | 2.2 [1.8, 2.6] | 30.54 [25.93, 35.76] | 37.1 [32.3, 42.3] | 15.6 [9.1, 22.8] |
| 4 | found | group:R | 2.2 [1.8, 2.6] | 31.05 [26.01, 37.05] | 35.5 [31.0, 40.6] | 30.8 [22.6, 39.4] |
| 4 | found | group:K | 3.7 [2.6, 4.8] | 51.23 [36.18, 67.75] | 9.1 [6.5, 12.1] | 36.1 [21.2, 51.6] |
| 4 | top1 | any | 2.2 [1.9, 2.7] | 30.16 [25.08, 36.23] | - | 15.3 [8.7, 22.7] |
| 4 | top1 | group:R | 2.2 [1.8, 2.6] | 29.20 [24.08, 35.51] | - | 30.8 [21.3, 40.5] |
| 4 | top1 | group:K | 4.2 [2.8, 5.7] | 56.48 [38.87, 75.63] | - | 46.4 [27.3, 65.0] |

## B1W hotpotqa (1000 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 0.7743 | 12.0 [10.0, 14.1] | 29.07 [25.51, 32.78] | POINTS | 9.5 [7.8, 11.3] ABOVE | 64.5 [58.3, 70.3] | 89.2 [84.3, 93.9] | 0.94937 | 0.92405 |
| label:slot | top1 | 0.759 | 19.0 [16.0, 22.5] | 42.49 [36.78, 49.75] | POINTS | 15.0 [12.3, 18.2] ABOVE | - | 88.8 [83.4, 93.7] | 0.95105 | 0.93007 |
| label:NER.spacy | found | 3.3122 | 2.4 [2.0, 2.8] | 5.76 [5.02, 6.55] | POINTS | -0.1 [-0.4, 0.1] LEVEL | 54.7 [48.2, 60.9] | 54.5 [45.8, 63.0] | 0.75373 | 0.58209 |
| label:NER.spacy | top1 | 3.549 | 3.7 [3.1, 4.3] | 8.33 [7.24, 9.48] | POINTS | -0.2 [-0.6, 0.2] LEVEL | - | 52.7 [43.7, 61.4] | 0.76744 | 0.55814 |
| label:K.knn | found | 2.2969 | 3.3 [2.7, 3.9] | 7.94 [6.91, 8.89] | POINTS | 0.8 [0.4, 1.1] ABOVE | 52.2 [45.7, 58.1] | 53.9 [44.8, 62.5] | 0.8125 | 0.53906 |
| label:K.knn | top1 | 2.333 | 5.2 [4.4, 6.1] | 11.71 [10.11, 13.33] | POINTS | 1.3 [0.7, 1.9] ABOVE | - | 53.8 [44.9, 62.6] | 0.78151 | 0.55462 |
| group:R | found | 3.5838 | 3.3 [2.8, 3.8] | 7.95 [7.26, 8.66] | POINTS | 0.8 [0.6, 1.0] ABOVE | 81.6 [76.6, 86.4] | 53.0 [46.1, 59.8] | 0.735 | 0.56 |
| group:R | top1 | 3.796 | 5.0 [4.3, 5.7] | 11.21 [10.11, 12.39] | POINTS | 1.1 [0.8, 1.4] ABOVE | - | 52.9 [45.8, 60.2] | 0.74332 | 0.54545 |
| group:K | found | 2.2969 | 3.3 [2.7, 3.9] | 7.94 [6.91, 8.89] | POINTS | 0.8 [0.4, 1.1] ABOVE | 52.2 [45.7, 58.1] | 53.9 [44.8, 62.5] | 0.8125 | 0.53906 |
| group:K | top1 | 2.333 | 5.2 [4.4, 6.1] | 11.71 [10.11, 13.33] | POINTS | 1.3 [0.7, 1.9] ABOVE | - | 53.8 [44.9, 62.6] | 0.78151 | 0.55462 |
| any | found | 5.0723 | 2.5 [2.1, 2.8] | 6.04 [5.62, 6.47] | POINTS | - | 87.8 [83.3, 91.7] | 36.3 [29.7, 42.5] | 0.6093 | 0.4093 |
| any | top1 | 5.266 | 3.9 [3.5, 4.4] | 8.80 [8.08, 9.55] | POINTS | - | - | 37.9 [31.4, 44.3] | 0.63547 | 0.40394 |

Multi-label minus single-label precision (found): 7.3 [5.9, 8.6] ABOVE (1774 / 6854 pairs).

Multi-label minus single-label precision (top1): 10.5 [8.6, 12.6] ABOVE (1101 / 4165 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | NER.spacy | 46.8 | 0.4 [0.2, 0.6] | 1.02 [0.59, 1.54] |
| found | K.knn | 29.3 | 0.6 [0.3, 0.9] | 1.44 [0.77, 2.20] |
| found | NER.spacy + K.knn | 8.6 | 3.4 [2.1, 4.7] | 8.19 [5.40, 11.36] |
| found | slot + NER.spacy + K.knn | 5.3 | 12.7 [9.4, 16.2] | 30.82 [23.62, 38.76] |
| found | slot + NER.spacy | 4.6 | 8.5 [5.7, 11.6] | 20.65 [14.15, 27.66] |
| found | slot | 3.3 | 12.7 [8.8, 17.2] | 30.82 [21.94, 40.97] |
| found | slot + K.knn | 2.1 | 16.8 [11.2, 22.9] | 40.61 [27.38, 55.06] |
| top1 | NER.spacy | 48.3 | 0.9 [0.6, 1.3] | 2.11 [1.35, 2.94] |
| top1 | K.knn | 27.9 | 1.2 [0.6, 1.7] | 2.59 [1.43, 3.83] |
| top1 | NER.spacy + K.knn | 9.3 | 4.5 [2.7, 6.3] | 10.04 [6.23, 14.05] |
| top1 | slot + NER.spacy + K.knn | 5.2 | 20.4 [15.5, 25.6] | 45.78 [35.57, 58.05] |
| top1 | slot + NER.spacy | 4.5 | 12.6 [8.3, 17.6] | 28.23 [18.84, 38.86] |
| top1 | slot | 2.8 | 20.8 [14.6, 28.3] | 46.60 [32.71, 63.35] |
| top1 | slot + K.knn | 1.9 | 27.6 [18.5, 37.8] | 61.71 [41.32, 85.12] |

## B1W 2wiki (1000 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 0.537 | 68.5 [65.1, 72.1] | 81.90 [77.29, 86.83] | POINTS | 60.2 [57.1, 63.5] ABOVE | 88.1 [85.4, 90.5] | 93.2 [91.3, 95.0] | 0.99026 | 0.93344 |
| label:slot | top1 | 0.594 | 72.1 [68.5, 76.0] | 85.92 [80.36, 91.86] | POINTS | 62.7 [59.3, 66.5] ABOVE | - | 93.0 [90.6, 95.3] | 0.98598 | 0.92523 |
| label:NER.spacy | found | 3.3214 | 9.6 [8.7, 10.6] | 11.48 [10.58, 12.50] | POINTS | 1.3 [0.9, 1.8] ABOVE | 75.9 [72.7, 79.1] | 56.6 [52.4, 60.6] | 0.76217 | 0.54494 |
| label:NER.spacy | top1 | 3.485 | 10.8 [9.8, 11.9] | 12.90 [11.77, 14.19] | POINTS | 1.5 [0.9, 2.1] ABOVE | - | 56.2 [51.1, 61.3] | 0.75332 | 0.54907 |
| label:K.knn | found | 2.2091 | 9.8 [9.0, 10.8] | 11.76 [10.87, 12.71] | POINTS | 1.5 [0.8, 2.3] ABOVE | 51.9 [48.0, 55.7] | 49.5 [44.2, 54.5] | 0.73901 | 0.43132 |
| label:K.knn | top1 | 2.395 | 12.1 [11.1, 13.2] | 14.44 [13.27, 15.71] | POINTS | 2.8 [1.8, 3.7] ABOVE | - | 47.9 [42.3, 53.4] | 0.74828 | 0.43448 |
| group:R | found | 3.4504 | 11.7 [10.6, 12.8] | 13.95 [12.98, 15.09] | POINTS | 3.4 [2.9, 3.9] ABOVE | 96.0 [94.3, 97.5] | 54.3 [50.6, 58.2] | 0.75223 | 0.52967 |
| group:R | top1 | 3.638 | 13.2 [12.1, 14.4] | 15.73 [14.52, 17.10] | POINTS | 3.8 [3.3, 4.5] ABOVE | - | 54.6 [50.1, 58.8] | 0.74375 | 0.53542 |
| group:K | found | 2.2091 | 9.8 [9.0, 10.8] | 11.76 [10.87, 12.71] | POINTS | 1.5 [0.8, 2.3] ABOVE | 51.9 [48.0, 55.7] | 49.5 [44.2, 54.5] | 0.73901 | 0.43132 |
| group:K | top1 | 2.395 | 12.1 [11.1, 13.2] | 14.44 [13.27, 15.71] | POINTS | 2.8 [1.8, 3.7] ABOVE | - | 47.9 [42.3, 53.4] | 0.74828 | 0.43448 |
| any | found | 4.9038 | 8.3 [7.7, 9.0] | 9.92 [9.41, 10.52] | POINTS | - | 96.8 [95.3, 98.2] | 33.3 [29.8, 37.2] | 0.54626 | 0.28781 |
| any | top1 | 5.184 | 9.4 [8.7, 10.1] | 11.16 [10.51, 11.88] | POINTS | - | - | 34.2 [30.1, 38.5] | 0.57938 | 0.30928 |

Multi-label minus single-label precision (found): 35.1 [32.7, 37.8] ABOVE (1555 / 6654 pairs).

Multi-label minus single-label precision (top1): 38.6 [35.8, 41.4] ABOVE (1009 / 4175 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | NER.spacy | 49.8 | 0.7 [0.4, 0.9] | 0.79 [0.50, 1.13] |
| found | K.knn | 29.6 | 0.3 [0.1, 0.5] | 0.34 [0.10, 0.63] |
| found | NER.spacy + K.knn | 9.6 | 3.9 [2.6, 5.4] | 4.70 [3.16, 6.44] |
| found | slot + NER.spacy + K.knn | 4.8 | 66.4 [61.7, 71.4] | 79.38 [73.16, 85.94] |
| found | slot + NER.spacy | 3.5 | 74.1 [68.2, 80.0] | 88.61 [81.52, 96.43] |
| found | slot | 1.6 | 56.4 [47.2, 66.7] | 67.40 [56.65, 79.78] |
| found | slot + K.knn | 1.0 | 78.3 [69.5, 86.8] | 93.60 [82.09, 105.04] |
| top1 | NER.spacy | 49.0 | 0.7 [0.4, 1.1] | 0.89 [0.52, 1.33] |
| top1 | K.knn | 29.8 | 0.3 [0.1, 0.6] | 0.39 [0.08, 0.75] |
| top1 | NER.spacy + K.knn | 9.8 | 6.5 [4.3, 8.7] | 7.78 [5.14, 10.36] |
| top1 | slot + NER.spacy + K.knn | 5.4 | 71.9 [67.0, 77.5] | 85.72 [78.80, 93.55] |
| top1 | slot + NER.spacy | 3.1 | 76.9 [69.9, 83.6] | 91.67 [83.33, 100.77] |
| top1 | slot | 1.8 | 58.2 [47.1, 69.5] | 69.45 [56.64, 83.15] |
| top1 | slot + K.knn | 1.2 | 80.6 [70.4, 89.9] | 96.17 [83.34, 108.26] |

## D83E squad (11873 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 12.998 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:R.mention | top1 | 12.8395 | 0.2 [0.2, 0.2] | 1.10 [1.00, 1.21] | NO_LIFT | -0.0 [-0.0, -0.0] BELOW | - | 29.2 [24.1, 34.8] | 0.50542 | 0.32852 |
| label:R.name_comention | found | 3.1158 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:R.name_comention | top1 | 3.1262 | 0.4 [0.3, 0.5] | 2.41 [2.07, 2.78] | POINTS | 0.2 [0.1, 0.2] ABOVE | - | 45.3 [37.2, 53.3] | 0.61486 | 0.40541 |
| label:R.title_comention | found | 0.0571 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:R.title_comention | top1 | 0.0594 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -0.2 [-0.2, -0.2] BELOW | - | - | None | None |
| label:K.knn | found | 1.8693 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:K.knn | top1 | 1.9592 | 0.4 [0.3, 0.5] | 2.60 [2.13, 3.10] | POINTS | 0.2 [0.1, 0.3] ABOVE | - | 66.0 [56.5, 75.0] | 0.87 | 0.61 |
| group:R | found | 13.9168 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| group:R | top1 | 13.805 | 0.2 [0.2, 0.2] | 1.23 [1.13, 1.34] | POINTS | -0.0 [-0.0, -0.0] BELOW | - | 28.7 [24.0, 33.7] | 0.48204 | 0.31437 |
| group:K | found | 1.8693 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| group:K | top1 | 1.9592 | 0.4 [0.3, 0.5] | 2.60 [2.13, 3.10] | POINTS | 0.2 [0.1, 0.3] ABOVE | - | 66.0 [56.5, 75.0] | 0.87 | 0.61 |
| any | found | 14.2827 | 0.0 [0.0, 0.0] | - | None | - | - | - | None | None |
| any | top1 | 14.1959 | 0.2 [0.2, 0.2] | 1.26 [1.17, 1.37] | POINTS | - | - | 28.7 [24.1, 33.5] | 0.47727 | 0.30682 |

Multi-label minus single-label precision (found): 0.0 [0.0, 0.0] LEVEL (35003 / 118536 pairs).

Multi-label minus single-label precision (top1): 0.2 [0.1, 0.3] ABOVE (38820 / 129728 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.mention | 68.7 | 0.0 [0.0, 0.0] | - |
| found | R.mention + R.name_comention | 11.9 | 0.0 [0.0, 0.0] | - |
| found | R.mention + K.knn | 6.5 | 0.0 [0.0, 0.0] | - |
| found | R.name_comention | 5.9 | 0.0 [0.0, 0.0] | - |
| found | R.mention + R.name_comention + K.knn | 3.5 | 0.0 [0.0, 0.0] | - |
| found | K.knn | 2.6 | 0.0 [0.0, 0.0] | - |
| top1 | R.mention | 68.0 | 0.1 [0.1, 0.1] | 0.76 [0.65, 0.87] |
| top1 | R.mention + R.name_comention | 11.6 | 0.3 [0.2, 0.4] | 1.73 [1.31, 2.19] |
| top1 | R.mention + K.knn | 6.8 | 0.4 [0.3, 0.5] | 2.28 [1.64, 2.96] |
| top1 | R.name_comention | 6.2 | 0.5 [0.4, 0.7] | 3.08 [2.31, 3.94] |
| top1 | R.mention + R.name_comention + K.knn | 3.6 | 0.6 [0.4, 0.8] | 3.48 [2.45, 4.71] |
| top1 | K.knn | 2.8 | 0.4 [0.2, 0.6] | 2.35 [1.37, 3.49] |
| top1 | R.name_comention + K.knn | 0.6 | 0.4 [0.0, 0.8] | 2.35 [0.00, 5.02] |

## D83E hotpotqa (7405 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 17.7544 | 1.4 [1.4, 1.5] | 19.14 [18.31, 20.02] | POINTS | 0.5 [0.4, 0.5] ABOVE | 70.7 [69.2, 72.2] | 51.9 [49.9, 53.9] | 0.69237 | 0.57827 |
| label:R.mention | top1 | 14.6631 | 2.2 [2.1, 2.3] | 26.24 [24.98, 27.53] | POINTS | 0.8 [0.8, 0.9] ABOVE | - | 51.5 [49.5, 53.4] | 0.68826 | 0.57131 |
| label:R.name_comention | found | 11.1538 | 0.8 [0.8, 0.9] | 11.37 [10.70, 12.00] | POINTS | -0.1 [-0.1, -0.1] BELOW | 26.4 [24.8, 27.9] | 45.4 [42.2, 48.5] | 0.6023 | 0.41858 |
| label:R.name_comention | top1 | 11.2533 | 1.1 [1.1, 1.2] | 13.51 [12.72, 14.29] | POINTS | -0.2 [-0.3, -0.2] BELOW | - | 43.7 [40.5, 46.8] | 0.58906 | 0.40343 |
| label:R.title_comention | found | 0.8891 | 5.0 [4.6, 5.5] | 67.72 [61.76, 74.10] | POINTS | 4.1 [3.7, 4.6] ABOVE | 12.5 [11.5, 13.6] | 87.5 [84.3, 90.5] | 0.94286 | 0.83516 |
| label:R.title_comention | top1 | 0.7683 | 7.2 [6.5, 8.0] | 85.83 [77.54, 94.85] | POINTS | 5.8 [5.1, 6.6] ABOVE | - | 88.5 [85.2, 91.6] | 0.95366 | 0.84878 |
| label:K.knn | found | 4.6882 | 1.2 [1.1, 1.3] | 16.09 [14.87, 17.35] | POINTS | 0.3 [0.2, 0.3] ABOVE | 15.7 [14.5, 16.9] | 47.0 [42.9, 51.1] | 0.74912 | 0.42632 |
| label:K.knn | top1 | 4.3733 | 1.8 [1.6, 1.9] | 20.92 [19.26, 22.61] | POINTS | 0.4 [0.3, 0.5] ABOVE | - | 45.8 [41.6, 49.9] | 0.73097 | 0.40177 |
| group:R | found | 26.4367 | 1.0 [1.0, 1.1] | 13.73 [13.29, 14.19] | POINTS | 0.1 [0.1, 0.1] ABOVE | 75.5 [74.1, 76.9] | 36.9 [35.0, 38.7] | 0.52881 | 0.38585 |
| group:R | top1 | 23.8116 | 1.5 [1.4, 1.5] | 17.59 [17.00, 18.24] | POINTS | 0.1 [0.1, 0.1] ABOVE | - | 36.1 [34.2, 38.0] | 0.52409 | 0.37685 |
| group:K | found | 4.6882 | 1.2 [1.1, 1.3] | 16.09 [14.87, 17.35] | POINTS | 0.3 [0.2, 0.3] ABOVE | 15.7 [14.5, 16.9] | 47.0 [42.9, 51.1] | 0.74912 | 0.42632 |
| group:K | top1 | 4.3733 | 1.8 [1.6, 1.9] | 20.92 [19.26, 22.61] | POINTS | 0.4 [0.3, 0.5] ABOVE | - | 45.8 [41.6, 49.9] | 0.73097 | 0.40177 |
| any | found | 29.3849 | 0.9 [0.9, 1.0] | 12.72 [12.34, 13.12] | POINTS | - | 77.8 [76.4, 79.1] | 24.3 [22.7, 25.9] | 0.38584 | 0.22796 |
| any | top1 | 26.728 | 1.4 [1.3, 1.4] | 16.25 [15.73, 16.80] | POINTS | - | - | 23.3 [21.7, 24.9] | 0.37936 | 0.21313 |

Multi-label minus single-label precision (found): 2.6 [2.4, 2.7] ABOVE (44508 / 253425 pairs).

Multi-label minus single-label precision (top1): 3.8 [3.6, 4.1] ABOVE (27794 / 170127 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.mention | 46.9 | 0.9 [0.8, 0.9] | 11.85 [11.13, 12.61] |
| found | R.name_comention | 27.7 | 0.1 [0.1, 0.1] | 1.64 [1.34, 1.96] |
| found | K.knn | 10.0 | 0.3 [0.2, 0.3] | 3.73 [2.96, 4.51] |
| found | R.mention + R.name_comention | 6.8 | 2.7 [2.5, 2.9] | 36.34 [33.34, 39.37] |
| found | R.mention + R.title_comention | 2.2 | 5.4 [4.8, 6.0] | 71.84 [64.44, 80.05] |
| found | R.mention + K.knn | 2.1 | 1.5 [1.2, 1.8] | 20.42 [16.56, 24.77] |
| found | R.mention + R.name_comention + K.knn | 2.0 | 4.2 [3.7, 4.7] | 56.69 [50.02, 63.60] |
| found | R.name_comention + K.knn | 1.4 | 1.2 [0.9, 1.6] | 16.77 [12.47, 21.54] |
| top1 | R.mention | 42.5 | 1.4 [1.3, 1.5] | 16.43 [15.30, 17.58] |
| top1 | R.name_comention | 32.1 | 0.2 [0.2, 0.2] | 2.49 [2.07, 2.93] |
| top1 | K.knn | 10.9 | 0.4 [0.4, 0.5] | 5.29 [4.23, 6.34] |
| top1 | R.mention + R.name_comention | 6.5 | 3.9 [3.6, 4.2] | 46.35 [42.49, 50.39] |
| top1 | R.mention + R.title_comention | 2.1 | 7.8 [6.9, 8.7] | 92.76 [82.90, 103.72] |
| top1 | R.mention + R.name_comention + K.knn | 1.8 | 6.9 [6.0, 7.8] | 82.37 [72.02, 92.95] |
| top1 | R.name_comention + K.knn | 1.7 | 1.8 [1.4, 2.3] | 21.41 [16.20, 26.98] |
| top1 | R.mention + K.knn | 1.6 | 2.8 [2.2, 3.4] | 33.42 [26.48, 40.95] |

## D83E 2wiki (12576 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 17.4847 | 2.9 [2.8, 2.9] | 31.63 [30.98, 32.31] | POINTS | 1.5 [1.4, 1.5] ABOVE | 73.7 [72.8, 74.6] | 26.4 [25.4, 27.3] | 0.40651 | 0.38306 |
| label:R.mention | top1 | 15.9251 | 3.2 [3.1, 3.3] | 34.69 [34.00, 35.37] | POINTS | 1.6 [1.6, 1.7] ABOVE | - | 28.9 [27.8, 29.9] | 0.43832 | 0.39424 |
| label:R.name_comention | found | 19.9165 | 1.3 [1.3, 1.3] | 14.45 [14.10, 14.84] | POINTS | -0.1 [-0.1, -0.1] BELOW | 38.3 [37.5, 39.3] | 22.3 [21.1, 23.6] | 0.36287 | 0.23286 |
| label:R.name_comention | top1 | 19.178 | 1.5 [1.4, 1.5] | 16.09 [15.63, 16.58] | POINTS | -0.1 [-0.1, -0.0] BELOW | - | 22.9 [21.5, 24.2] | 0.37451 | 0.24484 |
| label:R.title_comention | found | 0.57 | 8.6 [7.9, 9.2] | 94.25 [87.46, 101.15] | POINTS | 7.1 [6.5, 7.8] ABOVE | 7.2 [6.7, 7.6] | 83.4 [80.9, 86.0] | 0.96651 | 0.86721 |
| label:R.title_comention | top1 | 0.5653 | 9.5 [8.7, 10.2] | 102.24 [94.51, 110.15] | POINTS | 7.9 [7.2, 8.6] ABOVE | - | 83.9 [81.1, 86.6] | 0.9628 | 0.86607 |
| label:K.knn | found | 4.0962 | 2.5 [2.4, 2.6] | 27.19 [26.03, 28.49] | POINTS | 1.0 [0.9, 1.2] ABOVE | 14.8 [14.2, 15.5] | 39.6 [37.2, 41.8] | 0.65404 | 0.34596 |
| label:K.knn | top1 | 3.9552 | 3.0 [2.9, 3.2] | 32.53 [30.96, 34.11] | POINTS | 1.4 [1.3, 1.6] ABOVE | - | 39.3 [36.8, 41.8] | 0.65395 | 0.35676 |
| group:R | found | 34.775 | 1.5 [1.5, 1.6] | 16.82 [16.56, 17.10] | POINTS | 0.1 [0.1, 0.1] ABOVE | 77.9 [77.1, 78.8] | 15.7 [15.0, 16.5] | 0.26384 | 0.19881 |
| group:R | top1 | 32.6609 | 1.7 [1.6, 1.7] | 18.17 [17.85, 18.49] | POINTS | 0.1 [0.1, 0.1] ABOVE | - | 16.8 [15.9, 17.6] | 0.28062 | 0.20923 |
| group:K | found | 4.0962 | 2.5 [2.4, 2.6] | 27.19 [26.03, 28.49] | POINTS | 1.0 [0.9, 1.2] ABOVE | 14.8 [14.2, 15.5] | 39.6 [37.2, 41.8] | 0.65404 | 0.34596 |
| group:K | top1 | 3.9552 | 3.0 [2.9, 3.2] | 32.53 [30.96, 34.11] | POINTS | 1.4 [1.3, 1.6] ABOVE | - | 39.3 [36.8, 41.8] | 0.65395 | 0.35676 |
| any | found | 37.6746 | 1.4 [1.4, 1.5] | 15.64 [15.41, 15.89] | POINTS | - | 78.5 [77.7, 79.4] | 10.5 [9.8, 11.1] | 0.19141 | 0.09728 |
| any | top1 | 35.4587 | 1.6 [1.5, 1.6] | 16.95 [16.67, 17.23] | POINTS | - | - | 11.3 [10.6, 12.0] | 0.20829 | 0.1088 |

Multi-label minus single-label precision (found): 7.2 [7.0, 7.4] ABOVE (69156 / 599153 pairs).

Multi-label minus single-label precision (top1): 8.2 [8.0, 8.5] ABOVE (46563 / 399365 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.name_comention | 44.1 | 0.1 [0.1, 0.1] | 1.27 [1.14, 1.41] |
| found | R.mention | 37.5 | 1.4 [1.4, 1.5] | 15.90 [15.36, 16.41] |
| found | K.knn | 7.7 | 0.1 [0.1, 0.2] | 1.54 [1.19, 1.89] |
| found | R.mention + R.name_comention | 6.2 | 7.4 [7.2, 7.7] | 81.78 [79.08, 84.82] |
| found | R.name_comention + K.knn | 1.4 | 1.4 [1.1, 1.6] | 15.02 [12.48, 17.53] |
| found | R.mention + R.name_comention + K.knn | 1.2 | 13.8 [13.1, 14.7] | 152.28 [143.98, 161.11] |
| found | R.mention + R.title_comention | 1.0 | 9.6 [8.8, 10.3] | 105.24 [96.72, 113.88] |
| top1 | R.name_comention | 45.2 | 0.2 [0.1, 0.2] | 1.68 [1.50, 1.86] |
| top1 | R.mention | 36.1 | 1.5 [1.4, 1.5] | 16.03 [15.42, 16.64] |
| top1 | K.knn | 7.9 | 0.2 [0.2, 0.3] | 2.71 [2.15, 3.28] |
| top1 | R.mention + R.name_comention | 6.1 | 8.3 [8.0, 8.6] | 89.69 [86.06, 93.17] |
| top1 | R.name_comention + K.knn | 1.6 | 1.8 [1.5, 2.1] | 19.08 [15.89, 22.43] |
| top1 | R.mention + R.name_comention + K.knn | 1.2 | 16.5 [15.6, 17.6] | 178.83 [168.65, 189.91] |
| top1 | R.mention + R.title_comention | 1.0 | 10.6 [9.7, 11.6] | 114.76 [104.47, 125.29] |

## D83E musique (2417 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:R.mention | found | 38.7509 | 0.5 [0.5, 0.6] | 22.56 [20.09, 25.33] | POINTS | -0.2 [-0.2, -0.1] BELOW | 21.9 [20.2, 23.5] | 44.9 [40.9, 49.0] | 0.58741 | 0.4965 |
| label:R.mention | top1 | 46.5014 | 0.5 [0.5, 0.6] | 19.86 [17.65, 22.52] | POINTS | -0.2 [-0.2, -0.1] BELOW | - | 45.4 [41.4, 49.5] | 0.57366 | 0.48354 |
| label:R.name_comention | found | 17.3753 | 1.6 [1.5, 1.7] | 66.88 [62.39, 71.50] | POINTS | 0.9 [0.8, 1.0] ABOVE | 30.0 [28.2, 31.8] | 30.2 [27.0, 33.5] | 0.42951 | 0.32938 |
| label:R.name_comention | top1 | 18.3041 | 1.6 [1.5, 1.8] | 60.60 [56.05, 65.07] | POINTS | 0.9 [0.8, 1.0] ABOVE | - | 32.4 [28.9, 36.2] | 0.46165 | 0.36179 |
| label:R.title_comention | found | 0.5691 | 4.9 [3.7, 6.2] | 210.33 [160.57, 264.73] | POINTS | 4.2 [3.0, 5.5] ABOVE | 3.1 [2.4, 3.8] | 86.1 [78.3, 93.5] | 0.98734 | 0.88608 |
| label:R.title_comention | top1 | 0.6599 | 6.5 [5.2, 7.9] | 239.78 [194.09, 293.18] | POINTS | 5.8 [4.5, 7.2] ABOVE | - | 78.6 [70.6, 86.2] | 0.8932 | 0.81553 |
| label:K.knn | found | 4.0081 | 2.2 [1.9, 2.4] | 92.61 [81.74, 104.54] | POINTS | 1.5 [1.2, 1.7] ABOVE | 9.6 [8.5, 10.8] | 35.9 [30.0, 42.1] | 0.62041 | 0.37551 |
| label:K.knn | top1 | 4.103 | 2.9 [2.6, 3.3] | 109.33 [97.65, 121.40] | POINTS | 2.2 [1.9, 2.5] ABOVE | - | 43.1 [37.2, 48.4] | 0.64828 | 0.44828 |
| group:R | found | 53.8534 | 0.7 [0.6, 0.8] | 29.91 [27.43, 32.53] | POINTS | 0.0 [0.0, 0.0] ABOVE | 40.3 [38.3, 42.4] | 23.6 [21.2, 26.3] | 0.36416 | 0.27645 |
| group:R | top1 | 62.266 | 0.7 [0.6, 0.8] | 26.05 [23.83, 28.55] | POINTS | 0.0 [-0.0, 0.0] LEVEL | - | 26.7 [24.0, 29.4] | 0.38693 | 0.30251 |
| group:K | found | 4.0081 | 2.2 [1.9, 2.4] | 92.61 [81.74, 104.54] | POINTS | 1.5 [1.2, 1.7] ABOVE | 9.6 [8.5, 10.8] | 35.9 [30.0, 42.1] | 0.62041 | 0.37551 |
| group:K | top1 | 4.103 | 2.9 [2.6, 3.3] | 109.33 [97.65, 121.40] | POINTS | 2.2 [1.9, 2.5] ABOVE | - | 43.1 [37.2, 48.4] | 0.64828 | 0.44828 |
| any | found | 56.2748 | 0.7 [0.6, 0.7] | 29.10 [26.72, 31.60] | POINTS | - | 41.0 [39.1, 43.0] | 16.3 [14.0, 18.6] | 0.27554 | 0.17713 |
| any | top1 | 64.7385 | 0.7 [0.6, 0.8] | 26.03 [23.91, 28.41] | POINTS | - | - | 19.5 [17.1, 21.9] | 0.30971 | 0.20777 |

Multi-label minus single-label precision (found): 3.6 [3.1, 4.0] ABOVE (11196 / 148512 pairs).

Multi-label minus single-label precision (top1): 4.1 [3.7, 4.6] ABOVE (10364 / 146109 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | R.mention | 63.2 | 0.2 [0.2, 0.2] | 8.77 [7.48, 10.35] |
| found | R.name_comention | 25.1 | 1.0 [0.9, 1.1] | 42.00 [38.08, 46.21] |
| found | K.knn | 4.3 | 0.3 [0.1, 0.4] | 11.26 [6.41, 16.89] |
| found | R.mention + R.name_comention | 3.6 | 3.2 [2.8, 3.8] | 138.85 [118.42, 161.97] |
| found | R.name_comention + K.knn | 1.4 | 3.4 [2.6, 4.2] | 145.38 [112.81, 179.56] |
| found | R.mention + R.name_comention + K.knn | 0.8 | 9.2 [7.6, 11.0] | 396.48 [328.63, 469.98] |
| found | R.mention + K.knn | 0.6 | 1.1 [0.5, 1.9] | 49.22 [22.50, 81.33] |
| found | R.mention + R.title_comention | 0.6 | 3.5 [2.3, 5.0] | 152.08 [96.79, 216.55] |
| top1 | R.mention | 66.3 | 0.2 [0.2, 0.2] | 7.55 [6.37, 8.88] |
| top1 | R.name_comention | 22.9 | 1.0 [0.9, 1.1] | 36.66 [32.85, 40.59] |
| top1 | K.knn | 3.8 | 0.7 [0.5, 0.9] | 25.48 [18.26, 33.44] |
| top1 | R.mention + R.name_comention | 3.5 | 3.2 [2.7, 3.7] | 117.42 [99.75, 138.01] |
| top1 | R.name_comention + K.knn | 1.1 | 4.5 [3.6, 5.5] | 167.05 [133.09, 206.81] |
| top1 | R.mention + R.name_comention + K.knn | 0.8 | 9.6 [8.0, 11.3] | 357.93 [297.23, 420.28] |
| top1 | R.mention + R.title_comention | 0.6 | 5.2 [3.8, 6.7] | 192.13 [139.83, 249.06] |

| hops | kind | unit | precision | lift | reach | rank1 |
|---|---|---|---|---|---|---|
| 2 | found | any | 0.7 [0.6, 0.8] | 49.06 [44.11, 54.68] | 71.1 [68.0, 74.3] | 20.6 [17.4, 24.2] |
| 2 | found | group:R | 0.7 [0.6, 0.8] | 50.55 [45.32, 56.58] | 70.5 [67.4, 73.7] | 29.4 [25.7, 33.2] |
| 2 | found | group:K | 2.3 [1.9, 2.8] | 167.31 [140.26, 196.09] | 16.1 [13.5, 18.7] | 41.4 [33.1, 50.0] |
| 2 | top1 | any | 0.8 [0.7, 0.9] | 49.15 [43.49, 55.32] | - | 22.9 [19.6, 26.3] |
| 2 | top1 | group:R | 0.8 [0.7, 1.0] | 49.91 [44.04, 56.47] | - | 31.8 [28.2, 35.7] |
| 2 | top1 | group:K | 3.2 [2.7, 3.7] | 189.48 [162.51, 216.51] | - | 44.9 [37.3, 53.2] |
| 3 | found | any | 0.6 [0.5, 0.7] | 21.93 [19.15, 24.84] | 29.9 [27.2, 32.6] | 10.7 [7.2, 14.2] |
| 3 | found | group:R | 0.6 [0.5, 0.7] | 22.23 [19.27, 25.24] | 28.8 [26.2, 31.5] | 14.4 [10.5, 18.4] |
| 3 | found | group:K | 2.0 [1.6, 2.4] | 72.00 [57.15, 87.00] | 8.5 [6.8, 10.3] | 28.9 [19.2, 39.0] |
| 3 | top1 | any | 0.6 [0.5, 0.7] | 20.11 [17.36, 23.09] | - | 16.6 [12.5, 21.0] |
| 3 | top1 | group:R | 0.6 [0.5, 0.7] | 19.73 [16.93, 22.84] | - | 19.3 [14.9, 24.0] |
| 3 | top1 | group:K | 3.1 [2.5, 3.7] | 99.11 [81.28, 117.64] | - | 45.5 [35.0, 55.4] |
| 4 | found | any | 0.8 [0.6, 1.0] | 18.01 [14.71, 22.02] | 24.0 [21.1, 27.0] | 12.6 [7.9, 18.1] |
| 4 | found | group:R | 0.8 [0.7, 1.0] | 18.76 [15.23, 23.08] | 23.9 [21.0, 26.9] | 21.5 [15.4, 28.0] |
| 4 | found | group:K | 1.9 [1.3, 2.5] | 43.85 [30.14, 59.19] | 4.3 [3.0, 5.8] | 32.4 [16.7, 48.6] |
| 4 | top1 | any | 0.5 [0.4, 0.6] | 10.74 [8.46, 13.16] | - | 11.9 [7.0, 17.2] |
| 4 | top1 | group:R | 0.5 [0.4, 0.6] | 10.58 [8.32, 13.06] | - | 21.0 [14.1, 27.9] |
| 4 | top1 | group:K | 2.0 [1.4, 2.6] | 40.87 [28.01, 54.61] | - | 27.3 [13.0, 42.9] |

## D83E metaqa (9785 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:S | found | 12.1346 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -1.4 [-2.4, -0.6] BELOW | 0.0 [0.0, 0.0] | - | None | None |
| label:S | top1 | 6.3867 | 9.4 [9.0, 9.8] | 56.00 [52.73, 59.56] | POINTS | 3.4 [3.2, 3.5] ABOVE | - | 51.8 [50.1, 53.5] | 0.67896 | 0.49241 |
| label:R.mention | found | 0.0 | - | - | None | - None | 0.0 [0.0, 0.0] | - | None | None |
| label:R.mention | top1 | 0.0 | - | - | None | - None | - | - | None | None |
| label:R.name_comention | found | 0.2564 | 10.0 [0.0, 28.6] | 38.37 [0.00, 112.83] | NO_LIFT | 8.6 [-0.9, 26.2] LEVEL | 0.3 [0.0, 0.7] | 33.3 [0.0, 100.0] | 1.0 | 0.33333 |
| label:R.name_comention | top1 | 0.6054 | 0.1 [0.0, 0.2] | 0.60 [0.10, 1.21] | NO_LIFT | -5.9 [-6.2, -5.7] BELOW | - | 20.0 [0.0, 66.7] | 0.8 | 0.2 |
| label:R.title_comention | found | 0.0 | - | - | None | - None | 0.0 [0.0, 0.0] | - | None | None |
| label:R.title_comention | top1 | 0.0 | - | - | None | - None | - | - | None | None |
| label:K.knn | found | 3.0897 | 6.8 [3.2, 11.4] | 26.27 [11.98, 48.06] | POINTS | 5.4 [2.5, 9.1] ABOVE | 1.7 [0.9, 3.0] | 68.2 [40.0, 87.5] | 0.90909 | 0.59091 |
| label:K.knn | top1 | 3.1629 | 0.2 [0.1, 0.3] | 1.21 [0.88, 1.60] | NO_LIFT | -5.8 [-6.1, -5.6] BELOW | - | 54.9 [41.7, 69.2] | 0.7451 | 0.54902 |
| group:S | found | 12.1346 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -1.4 [-2.4, -0.6] BELOW | 0.0 [0.0, 0.0] | - | None | None |
| group:S | top1 | 6.3867 | 9.4 [9.0, 9.8] | 56.00 [52.73, 59.56] | POINTS | 3.4 [3.2, 3.5] ABOVE | - | 51.8 [50.1, 53.5] | 0.67896 | 0.49241 |
| group:R | found | 0.2564 | 10.0 [0.0, 28.6] | 38.37 [0.00, 112.83] | NO_LIFT | 8.6 [-0.9, 26.2] LEVEL | 0.3 [0.0, 0.7] | 33.3 [0.0, 100.0] | 1.0 | 0.33333 |
| group:R | top1 | 0.6054 | 0.1 [0.0, 0.2] | 0.60 [0.10, 1.21] | NO_LIFT | -5.9 [-6.2, -5.7] BELOW | - | 20.0 [0.0, 66.7] | 0.8 | 0.2 |
| group:K | found | 3.0897 | 6.8 [3.2, 11.4] | 26.27 [11.98, 48.06] | POINTS | 5.4 [2.5, 9.1] ABOVE | 1.7 [0.9, 3.0] | 68.2 [40.0, 87.5] | 0.90909 | 0.59091 |
| group:K | top1 | 3.1629 | 0.2 [0.1, 0.3] | 1.21 [0.88, 1.60] | NO_LIFT | -5.8 [-6.1, -5.6] BELOW | - | 54.9 [41.7, 69.2] | 0.7451 | 0.54902 |
| any | found | 15.3141 | 1.4 [0.6, 2.4] | 5.46 [2.39, 9.87] | POINTS | - | 1.7 [0.9, 3.0] | 59.1 [27.3, 81.0] | 0.86364 | 0.54545 |
| any | top1 | 10.0494 | 6.0 [5.8, 6.3] | 35.99 [33.89, 38.22] | POINTS | - | - | 8.3 [7.3, 9.2] | 0.1938 | 0.11815 |

Multi-label minus single-label precision (found): 10.2 [-0.9, 24.7] LEVEL (26 / 2363 pairs).

Multi-label minus single-label precision (top1): -5.8 [-6.2, -5.3] BELOW (1031 / 97302 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | S | 78.7 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| found | K.knn | 19.1 | 6.6 [3.0, 11.0] | 25.24 [10.88, 46.27] |
| found | R.name_comention | 1.1 | 3.7 [0.0, 18.2] | 14.21 [0.00, 72.54] |
| found | S + K.knn | 0.5 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| top1 | S | 63.5 | 9.4 [9.0, 9.8] | 56.08 [52.79, 59.63] |
| top1 | K.knn | 30.4 | 0.2 [0.1, 0.3] | 1.20 [0.86, 1.59] |
| top1 | R.name_comention | 5.1 | 0.1 [0.0, 0.2] | 0.48 [0.00, 1.11] |
| top1 | R.name_comention + K.knn | 1.0 | 0.2 [0.0, 0.6] | 1.27 [0.00, 3.31] |

## D83E webqsp (1503 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:S | found | 158.6882 | 0.2 [0.0, 0.7] | 1.61 [0.20, 4.57] | NO_LIFT | -0.2 [-0.3, -0.1] BELOW | 6.5 [0.9, 17.6] | 23.1 [0.0, 50.0] | 0.23077 | 0.23077 |
| label:S | top1 | 58.0998 | 1.8 [1.3, 2.4] | 15.34 [11.40, 20.08] | POINTS | 0.1 [0.1, 0.2] ABOVE | - | 18.7 [15.5, 22.1] | 0.35689 | 0.24028 |
| label:R.mention | found | 0.0 | - | - | None | - None | 0.0 [0.0, 0.0] | - | None | None |
| label:R.mention | top1 | 0.0 | - | - | None | - None | - | - | None | None |
| label:R.name_comention | found | 1.3118 | 2.5 [0.0, 6.5] | 16.25 [0.00, 40.54] | NO_LIFT | 2.0 [-0.7, 6.0] LEVEL | 0.3 [0.0, 0.9] | 0.0 [0.0, 0.0] | 0.0 | 0.0 |
| label:R.name_comention | top1 | 1.656 | 1.1 [0.7, 1.6] | 9.62 [5.96, 13.78] | POINTS | -0.6 [-1.1, 0.0] LEVEL | - | 48.0 [27.8, 68.2] | 0.64 | 0.44 |
| label:R.title_comention | found | 0.0 | - | - | None | - None | 0.0 [0.0, 0.0] | - | None | None |
| label:R.title_comention | top1 | 0.0007 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -1.7 [-2.2, -1.3] BELOW | - | - | None | None |
| label:K.knn | found | 4.0484 | 6.8 [4.1, 9.6] | 44.75 [27.65, 68.69] | POINTS | 6.4 [3.7, 9.1] ABOVE | 4.0 [2.5, 6.0] | 71.4 [54.5, 86.4] | 0.91429 | 0.68571 |
| label:K.knn | top1 | 5.5669 | 0.9 [0.7, 1.2] | 7.97 [5.98, 10.40] | POINTS | -0.8 [-1.3, -0.3] BELOW | - | 36.9 [25.4, 48.6] | 0.6 | 0.43077 |
| group:S | found | 158.6882 | 0.2 [0.0, 0.7] | 1.61 [0.20, 4.57] | NO_LIFT | -0.2 [-0.3, -0.1] BELOW | 6.5 [0.9, 17.6] | 23.1 [0.0, 50.0] | 0.23077 | 0.23077 |
| group:S | top1 | 58.0998 | 1.8 [1.3, 2.4] | 15.34 [11.40, 20.08] | POINTS | 0.1 [0.1, 0.2] ABOVE | - | 18.7 [15.5, 22.1] | 0.35689 | 0.24028 |
| group:R | found | 1.3118 | 2.5 [0.0, 6.5] | 16.25 [0.00, 40.54] | NO_LIFT | 2.0 [-0.7, 6.0] LEVEL | 0.3 [0.0, 0.9] | 0.0 [0.0, 0.0] | 0.0 | 0.0 |
| group:R | top1 | 1.6567 | 1.1 [0.7, 1.6] | 9.61 [5.95, 13.78] | POINTS | -0.6 [-1.1, 0.0] LEVEL | - | 48.0 [27.8, 68.2] | 0.64 | 0.44 |
| group:K | found | 4.0484 | 6.8 [4.1, 9.6] | 44.75 [27.65, 68.69] | POINTS | 6.4 [3.7, 9.1] ABOVE | 4.0 [2.5, 6.0] | 71.4 [54.5, 86.4] | 0.91429 | 0.68571 |
| group:K | top1 | 5.5669 | 0.9 [0.7, 1.2] | 7.97 [5.98, 10.40] | POINTS | -0.8 [-1.3, -0.3] BELOW | - | 36.9 [25.4, 48.6] | 0.6 | 0.43077 |
| any | found | 162.8172 | 0.4 [0.2, 0.9] | 2.73 [1.08, 5.81] | POINTS | - | 10.4 [4.3, 21.4] | 41.3 [24.2, 59.1] | 0.52174 | 0.45652 |
| any | top1 | 64.0353 | 1.7 [1.3, 2.2] | 14.40 [10.90, 18.64] | POINTS | - | - | 6.7 [4.8, 8.7] | 0.13093 | 0.10475 |

Multi-label minus single-label precision (found): 1.5 [-0.6, 4.3] LEVEL (212 / 30072 pairs).

Multi-label minus single-label precision (top1): 0.9 [0.1, 1.8] ABOVE (1772 / 94473 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | S | 96.8 | 0.2 [0.0, 0.7] | 1.58 [0.19, 4.53] |
| found | K.knn | 1.9 | 8.1 [5.0, 11.6] | 53.55 [32.80, 83.56] |
| found | R.name_comention | 0.5 | 2.4 [0.0, 6.0] | 15.92 [0.00, 37.69] |
| top1 | S | 89.3 | 1.8 [1.3, 2.4] | 15.15 [11.24, 19.93] |
| top1 | K.knn | 7.1 | 0.6 [0.4, 0.8] | 4.89 [3.19, 6.93] |
| top1 | R.name_comention | 1.7 | 0.8 [0.4, 1.4] | 7.20 [3.24, 12.06] |
| top1 | S + K.knn | 1.0 | 3.3 [2.0, 4.7] | 28.07 [16.99, 40.66] |

## REC squad (11873 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 12.998 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:slot | top1 | 12.8395 | 0.2 [0.2, 0.2] | 1.10 [1.00, 1.21] | NO_LIFT | -0.0 [-0.0, -0.0] BELOW | - | 29.2 [24.1, 34.8] | 0.50542 | 0.32852 |
| label:NER.spacy | found | 2.9525 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:NER.spacy | top1 | 2.9941 | 0.4 [0.3, 0.4] | 2.30 [1.96, 2.67] | POINTS | 0.2 [0.1, 0.2] ABOVE | - | 38.5 [30.4, 46.7] | 0.6 | 0.39259 |
| label:K.knn | found | 1.8693 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| label:K.knn | top1 | 1.9592 | 0.4 [0.3, 0.5] | 2.60 [2.13, 3.10] | POINTS | 0.2 [0.1, 0.3] ABOVE | - | 66.0 [56.5, 75.0] | 0.87 | 0.61 |
| group:R | found | 13.7939 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| group:R | top1 | 13.6821 | 0.2 [0.2, 0.2] | 1.23 [1.13, 1.33] | POINTS | -0.0 [-0.0, -0.0] BELOW | - | 27.6 [22.8, 32.6] | 0.46667 | 0.30909 |
| group:K | found | 1.8693 | 0.0 [0.0, 0.0] | - | None | 0.0 [0.0, 0.0] LEVEL | - | - | None | None |
| group:K | top1 | 1.9592 | 0.4 [0.3, 0.5] | 2.60 [2.13, 3.10] | POINTS | 0.2 [0.1, 0.3] ABOVE | - | 66.0 [56.5, 75.0] | 0.87 | 0.61 |
| any | found | 14.1665 | 0.0 [0.0, 0.0] | - | None | - | - | - | None | None |
| any | top1 | 14.0797 | 0.2 [0.2, 0.2] | 1.26 [1.16, 1.37] | POINTS | - | - | 27.9 [23.2, 32.6] | 0.45977 | 0.30747 |

Multi-label minus single-label precision (found): 0.0 [0.0, 0.0] LEVEL (34291 / 117999 pairs).

Multi-label minus single-label precision (top1): 0.2 [0.1, 0.2] ABOVE (38387 / 128781 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | slot | 69.7 | 0.0 [0.0, 0.0] | - |
| found | slot + NER.spacy | 12.0 | 0.0 [0.0, 0.0] | - |
| found | slot + K.knn | 6.8 | 0.0 [0.0, 0.0] | - |
| found | NER.spacy | 5.1 | 0.0 [0.0, 0.0] | - |
| found | slot + NER.spacy + K.knn | 3.3 | 0.0 [0.0, 0.0] | - |
| found | K.knn | 2.6 | 0.0 [0.0, 0.0] | - |
| top1 | slot | 68.8 | 0.1 [0.1, 0.2] | 0.80 [0.68, 0.91] |
| top1 | slot + NER.spacy | 11.9 | 0.2 [0.2, 0.3] | 1.46 [1.08, 1.89] |
| top1 | slot + K.knn | 7.1 | 0.4 [0.3, 0.5] | 2.24 [1.60, 2.88] |
| top1 | NER.spacy | 5.4 | 0.5 [0.4, 0.7] | 3.28 [2.43, 4.19] |
| top1 | slot + NER.spacy + K.knn | 3.4 | 0.6 [0.4, 0.8] | 3.61 [2.49, 4.92] |
| top1 | K.knn | 2.8 | 0.4 [0.2, 0.6] | 2.31 [1.35, 3.44] |

## REC hotpotqa (7405 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 17.754 | 1.4 [1.4, 1.5] | 19.14 [18.31, 20.02] | POINTS | 0.5 [0.5, 0.6] ABOVE | 70.7 [69.2, 72.2] | 51.9 [49.9, 53.9] | 0.69237 | 0.57827 |
| label:slot | top1 | 14.6629 | 2.2 [2.1, 2.3] | 26.24 [24.98, 27.53] | POINTS | 0.9 [0.8, 1.0] ABOVE | - | 51.5 [49.5, 53.4] | 0.68826 | 0.57131 |
| label:NER.spacy | found | 12.0132 | 0.7 [0.6, 0.7] | 9.10 [8.52, 9.68] | POINTS | -0.2 [-0.3, -0.2] BELOW | 22.7 [21.3, 24.1] | 48.3 [44.9, 51.9] | 0.63196 | 0.43705 |
| label:NER.spacy | top1 | 12.071 | 0.9 [0.9, 1.0] | 11.03 [10.32, 11.72] | POINTS | -0.4 [-0.4, -0.3] BELOW | - | 47.2 [43.8, 50.5] | 0.63081 | 0.42787 |
| label:K.knn | found | 4.6896 | 1.2 [1.1, 1.3] | 16.09 [14.87, 17.35] | POINTS | 0.3 [0.2, 0.4] ABOVE | 15.7 [14.5, 16.9] | 47.0 [42.9, 51.1] | 0.74912 | 0.42632 |
| label:K.knn | top1 | 4.3741 | 1.8 [1.6, 1.9] | 20.92 [19.26, 22.60] | POINTS | 0.5 [0.3, 0.6] ABOVE | - | 45.8 [41.6, 49.9] | 0.73097 | 0.40177 |
| group:R | found | 27.5621 | 1.0 [0.9, 1.0] | 12.99 [12.57, 13.44] | POINTS | 0.1 [0.1, 0.1] ABOVE | 74.5 [73.0, 75.9] | 38.6 [36.7, 40.4] | 0.54935 | 0.4 |
| group:R | top1 | 24.8201 | 1.4 [1.3, 1.5] | 16.69 [16.12, 17.30] | POINTS | 0.1 [0.1, 0.1] ABOVE | - | 38.3 [36.4, 40.2] | 0.54642 | 0.393 |
| group:K | found | 4.6896 | 1.2 [1.1, 1.3] | 16.09 [14.87, 17.35] | POINTS | 0.3 [0.2, 0.4] ABOVE | 15.7 [14.5, 16.9] | 47.0 [42.9, 51.1] | 0.74912 | 0.42632 |
| group:K | top1 | 4.3741 | 1.8 [1.6, 1.9] | 20.92 [19.26, 22.60] | POINTS | 0.5 [0.3, 0.6] ABOVE | - | 45.8 [41.6, 49.9] | 0.73097 | 0.40177 |
| any | found | 30.546 | 0.9 [0.9, 0.9] | 12.13 [11.76, 12.50] | POINTS | - | 77.1 [75.6, 78.5] | 25.3 [23.7, 27.0] | 0.39871 | 0.22972 |
| any | top1 | 27.7803 | 1.3 [1.3, 1.4] | 15.52 [15.03, 16.04] | POINTS | - | - | 24.5 [22.8, 26.1] | 0.39516 | 0.21726 |

Multi-label minus single-label precision (found): 2.0 [1.8, 2.2] ABOVE (34464 / 275242 pairs).

Multi-label minus single-label precision (top1): 3.2 [2.9, 3.4] ABOVE (21527 / 184186 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | slot | 48.2 | 1.1 [1.1, 1.2] | 15.05 [14.24, 15.88] |
| found | NER.spacy | 30.9 | 0.1 [0.1, 0.1] | 1.32 [1.07, 1.59] |
| found | K.knn | 9.8 | 0.3 [0.2, 0.4] | 4.17 [3.33, 5.00] |
| found | slot + NER.spacy | 5.5 | 2.7 [2.4, 2.9] | 35.87 [32.66, 39.16] |
| found | slot + K.knn | 2.7 | 2.5 [2.1, 2.8] | 32.97 [28.65, 37.69] |
| found | slot + NER.spacy + K.knn | 1.7 | 4.4 [3.9, 5.0] | 59.49 [52.05, 67.03] |
| found | NER.spacy + K.knn | 1.2 | 1.1 [0.8, 1.5] | 15.01 [10.82, 19.48] |
| top1 | slot | 43.8 | 1.7 [1.6, 1.8] | 20.52 [19.30, 21.79] |
| top1 | NER.spacy | 35.1 | 0.2 [0.2, 0.2] | 2.16 [1.81, 2.55] |
| top1 | K.knn | 10.7 | 0.5 [0.4, 0.6] | 5.70 [4.63, 6.79] |
| top1 | slot + NER.spacy | 5.4 | 3.9 [3.5, 4.3] | 46.21 [41.83, 50.62] |
| top1 | slot + K.knn | 2.1 | 4.5 [3.9, 5.2] | 53.97 [46.31, 61.86] |
| top1 | slot + NER.spacy + K.knn | 1.5 | 6.9 [6.0, 7.9] | 82.20 [71.39, 93.38] |
| top1 | NER.spacy + K.knn | 1.5 | 1.7 [1.3, 2.2] | 20.83 [15.56, 26.57] |

## REC 2wiki (12576 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 17.4843 | 2.9 [2.8, 2.9] | 31.63 [30.98, 32.31] | POINTS | 1.5 [1.5, 1.6] ABOVE | 73.7 [72.8, 74.6] | 26.4 [25.4, 27.3] | 0.40651 | 0.38306 |
| label:slot | top1 | 15.9248 | 3.2 [3.1, 3.3] | 34.69 [34.00, 35.37] | POINTS | 1.7 [1.7, 1.8] ABOVE | - | 28.9 [27.8, 29.9] | 0.43832 | 0.39424 |
| label:NER.spacy | found | 21.5237 | 1.1 [1.1, 1.1] | 12.07 [11.77, 12.41] | POINTS | -0.2 [-0.3, -0.2] BELOW | 34.6 [33.9, 35.5] | 24.4 [23.1, 25.8] | 0.38658 | 0.25621 |
| label:NER.spacy | top1 | 20.4801 | 1.2 [1.2, 1.3] | 13.50 [13.10, 13.92] | POINTS | -0.2 [-0.3, -0.2] BELOW | - | 25.1 [23.7, 26.7] | 0.40068 | 0.27117 |
| label:K.knn | found | 4.0966 | 2.5 [2.4, 2.6] | 27.18 [26.02, 28.49] | POINTS | 1.1 [1.0, 1.2] ABOVE | 14.8 [14.2, 15.5] | 39.6 [37.2, 41.8] | 0.65404 | 0.34596 |
| label:K.knn | top1 | 3.9552 | 3.0 [2.9, 3.2] | 32.53 [30.96, 34.11] | POINTS | 1.5 [1.4, 1.7] ABOVE | - | 39.3 [36.8, 41.8] | 0.65395 | 0.35676 |
| group:R | found | 36.6513 | 1.4 [1.4, 1.5] | 15.79 [15.54, 16.06] | POINTS | 0.1 [0.1, 0.1] ABOVE | 77.1 [76.3, 78.0] | 17.3 [16.5, 18.1] | 0.28205 | 0.20958 |
| group:R | top1 | 34.2312 | 1.6 [1.6, 1.6] | 17.10 [16.80, 17.41] | POINTS | 0.1 [0.1, 0.1] ABOVE | - | 18.6 [17.7, 19.5] | 0.3004 | 0.22379 |
| group:K | found | 4.0966 | 2.5 [2.4, 2.6] | 27.18 [26.02, 28.49] | POINTS | 1.1 [1.0, 1.2] ABOVE | 14.8 [14.2, 15.5] | 39.6 [37.2, 41.8] | 0.65404 | 0.34596 |
| group:K | top1 | 3.9552 | 3.0 [2.9, 3.2] | 32.53 [30.96, 34.11] | POINTS | 1.5 [1.4, 1.7] ABOVE | - | 39.3 [36.8, 41.8] | 0.65395 | 0.35676 |
| any | found | 39.6176 | 1.3 [1.3, 1.4] | 14.75 [14.52, 15.00] | POINTS | - | 77.9 [77.0, 78.8] | 11.0 [10.3, 11.7] | 0.20219 | 0.10178 |
| any | top1 | 37.1097 | 1.5 [1.5, 1.5] | 16.03 [15.77, 16.31] | POINTS | - | - | 11.9 [11.2, 12.6] | 0.21992 | 0.11496 |

Multi-label minus single-label precision (found): 7.5 [7.3, 7.8] ABOVE (54896 / 647881 pairs).

Multi-label minus single-label precision (top1): 8.8 [8.5, 9.1] ABOVE (36333 / 430359 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | NER.spacy | 47.2 | 0.1 [0.1, 0.1] | 1.00 [0.89, 1.12] |
| found | slot | 37.5 | 1.7 [1.7, 1.8] | 18.79 [18.25, 19.35] |
| found | K.knn | 7.5 | 0.2 [0.1, 0.2] | 1.97 [1.58, 2.36] |
| found | slot + NER.spacy | 5.0 | 8.1 [7.8, 8.4] | 89.16 [86.02, 92.54] |
| found | NER.spacy + K.knn | 1.2 | 1.3 [1.1, 1.5] | 14.56 [11.95, 17.06] |
| found | slot + NER.spacy + K.knn | 1.0 | 13.7 [12.9, 14.5] | 150.53 [141.68, 159.75] |
| top1 | NER.spacy | 48.0 | 0.1 [0.1, 0.1] | 1.33 [1.18, 1.49] |
| top1 | slot | 36.4 | 1.8 [1.7, 1.9] | 19.42 [18.79, 20.07] |
| top1 | K.knn | 7.8 | 0.3 [0.3, 0.4] | 3.32 [2.72, 3.94] |
| top1 | slot + NER.spacy | 4.9 | 9.2 [8.8, 9.6] | 99.22 [95.07, 103.36] |
| top1 | NER.spacy + K.knn | 1.3 | 1.7 [1.4, 2.1] | 18.77 [15.21, 22.38] |
| top1 | slot + NER.spacy + K.knn | 1.0 | 16.3 [15.2, 17.4] | 176.41 [164.84, 188.83] |

## REC musique (2417 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 38.5032 | 0.5 [0.5, 0.6] | 22.55 [20.01, 25.33] | POINTS | -0.1 [-0.1, -0.1] BELOW | 21.7 [20.0, 23.4] | 44.9 [40.8, 49.0] | 0.58803 | 0.49824 |
| label:slot | top1 | 46.1746 | 0.5 [0.5, 0.6] | 19.56 [17.37, 22.17] | POINTS | -0.1 [-0.2, -0.1] BELOW | - | 46.1 [42.0, 50.2] | 0.58156 | 0.49291 |
| label:NER.spacy | found | 16.9922 | 1.5 [1.4, 1.6] | 64.11 [59.68, 68.82] | POINTS | 0.8 [0.8, 0.9] ABOVE | 28.1 [26.3, 30.0] | 36.9 [33.5, 40.3] | 0.50559 | 0.39944 |
| label:NER.spacy | top1 | 17.964 | 1.5 [1.4, 1.6] | 56.61 [52.43, 61.26] | POINTS | 0.9 [0.8, 1.0] ABOVE | - | 39.4 [35.5, 43.1] | 0.52492 | 0.419 |
| label:K.knn | found | 4.0617 | 2.1 [1.9, 2.4] | 91.39 [80.56, 103.23] | POINTS | 1.5 [1.2, 1.7] ABOVE | 9.6 [8.5, 10.8] | 35.5 [29.6, 41.6] | 0.61633 | 0.37143 |
| label:K.knn | top1 | 4.1804 | 2.9 [2.6, 3.2] | 107.31 [95.78, 119.13] | POINTS | 2.2 [1.9, 2.5] ABOVE | - | 42.8 [36.9, 48.1] | 0.64483 | 0.43793 |
| group:R | found | 53.5965 | 0.7 [0.6, 0.7] | 28.21 [25.88, 30.70] | POINTS | 0.0 [0.0, 0.0] ABOVE | 37.7 [35.9, 39.7] | 27.7 [25.0, 30.6] | 0.39232 | 0.31446 |
| group:R | top1 | 62.0008 | 0.7 [0.6, 0.7] | 24.18 [22.06, 26.62] | POINTS | -0.0 [-0.0, 0.0] LEVEL | - | 30.0 [27.1, 33.0] | 0.40713 | 0.33261 |
| group:K | found | 4.0617 | 2.1 [1.9, 2.4] | 91.39 [80.56, 103.23] | POINTS | 1.5 [1.2, 1.7] ABOVE | 9.6 [8.5, 10.8] | 35.5 [29.6, 41.6] | 0.61633 | 0.37143 |
| group:K | top1 | 4.1804 | 2.9 [2.6, 3.2] | 107.31 [95.78, 119.13] | POINTS | 2.2 [1.9, 2.5] ABOVE | - | 42.8 [36.9, 48.1] | 0.64483 | 0.43793 |
| any | found | 56.1952 | 0.6 [0.6, 0.7] | 27.66 [25.47, 29.96] | POINTS | - | 38.8 [37.0, 40.8] | 17.6 [15.3, 20.0] | 0.29597 | 0.18486 |
| any | top1 | 64.6574 | 0.7 [0.6, 0.7] | 24.52 [22.47, 26.84] | POINTS | - | - | 20.1 [17.6, 22.6] | 0.32446 | 0.20983 |

Multi-label minus single-label precision (found): 4.4 [3.9, 4.9] ABOVE (8502 / 150980 pairs).

Multi-label minus single-label precision (top1): 4.7 [4.3, 5.3] ABOVE (7892 / 148385 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | slot | 64.4 | 0.2 [0.2, 0.3] | 9.80 [8.38, 11.43] |
| found | NER.spacy | 25.7 | 0.9 [0.8, 1.0] | 37.58 [34.10, 41.33] |
| found | K.knn | 4.6 | 0.4 [0.2, 0.5] | 16.32 [10.51, 22.60] |
| found | slot + NER.spacy | 2.7 | 4.3 [3.7, 5.0] | 186.82 [160.32, 216.50] |
| found | NER.spacy + K.knn | 1.2 | 3.6 [2.7, 4.5] | 154.14 [118.21, 194.26] |
| found | slot + K.knn | 0.8 | 3.6 [2.6, 4.7] | 155.54 [114.90, 203.09] |
| top1 | slot | 67.4 | 0.2 [0.2, 0.3] | 9.09 [7.82, 10.53] |
| top1 | NER.spacy | 23.4 | 0.8 [0.8, 0.9] | 31.46 [27.97, 35.28] |
| top1 | K.knn | 4.1 | 0.9 [0.7, 1.1] | 32.38 [24.62, 40.59] |
| top1 | slot + NER.spacy | 2.7 | 4.1 [3.5, 4.8] | 151.74 [127.82, 177.44] |
| top1 | NER.spacy + K.knn | 1.1 | 4.7 [3.7, 5.7] | 174.58 [138.21, 212.19] |
| top1 | slot + K.knn | 0.7 | 5.3 [3.9, 6.7] | 195.06 [146.04, 248.85] |

| hops | kind | unit | precision | lift | reach | rank1 |
|---|---|---|---|---|---|---|
| 2 | found | any | 0.7 [0.6, 0.8] | 47.81 [42.98, 53.25] | 69.5 [66.3, 72.7] | 20.8 [17.5, 24.1] |
| 2 | found | group:R | 0.7 [0.6, 0.8] | 49.02 [43.92, 54.77] | 68.4 [65.2, 71.6] | 32.5 [28.8, 36.4] |
| 2 | found | group:K | 2.3 [1.9, 2.7] | 165.98 [139.03, 194.57] | 16.1 [13.5, 18.7] | 41.4 [33.1, 50.0] |
| 2 | top1 | any | 0.8 [0.7, 0.9] | 47.84 [42.39, 53.65] | - | 22.7 [19.4, 26.1] |
| 2 | top1 | group:R | 0.8 [0.7, 0.9] | 48.31 [42.70, 54.42] | - | 34.4 [30.6, 38.2] |
| 2 | top1 | group:K | 3.2 [2.7, 3.6] | 187.55 [160.77, 214.49] | - | 44.9 [37.3, 53.2] |
| 3 | found | any | 0.6 [0.5, 0.7] | 21.07 [18.26, 23.94] | 28.6 [25.8, 31.3] | 13.8 [9.9, 17.7] |
| 3 | found | group:R | 0.6 [0.5, 0.7] | 21.09 [18.24, 24.12] | 26.9 [24.2, 29.6] | 21.1 [16.3, 26.1] |
| 3 | found | group:K | 2.0 [1.6, 2.4] | 70.93 [56.32, 85.75] | 8.5 [6.8, 10.3] | 27.7 [18.1, 37.5] |
| 3 | top1 | any | 0.5 [0.5, 0.6] | 17.51 [15.09, 20.17] | - | 18.6 [13.9, 23.5] |
| 3 | top1 | group:R | 0.5 [0.4, 0.6] | 16.38 [13.94, 19.19] | - | 23.7 [18.2, 29.2] |
| 3 | top1 | group:K | 3.0 [2.5, 3.6] | 96.36 [78.99, 114.33] | - | 44.4 [34.3, 54.3] |
| 4 | found | any | 0.7 [0.5, 0.8] | 15.52 [12.51, 19.20] | 20.3 [17.6, 23.0] | 13.9 [8.7, 20.1] |
| 4 | found | group:R | 0.7 [0.6, 0.9] | 16.07 [12.87, 20.00] | 19.9 [17.3, 22.6] | 23.2 [17.0, 30.9] |
| 4 | found | group:K | 1.9 [1.3, 2.5] | 42.75 [29.36, 57.83] | 4.3 [3.0, 5.8] | 32.4 [16.7, 48.6] |
| 4 | top1 | any | 0.5 [0.4, 0.6] | 10.33 [8.21, 12.64] | - | 11.8 [6.6, 17.4] |
| 4 | top1 | group:R | 0.5 [0.4, 0.6] | 10.20 [8.04, 12.63] | - | 23.1 [16.0, 30.3] |
| 4 | top1 | group:K | 1.9 [1.3, 2.5] | 39.86 [27.27, 53.41] | - | 27.3 [13.0, 42.9] |

## REC metaqa (9785 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 12.1346 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -1.5 [-2.5, -0.7] BELOW | 0.0 [0.0, 0.0] | - | None | None |
| label:slot | top1 | 6.3867 | 9.4 [9.0, 9.8] | 56.00 [52.73, 59.56] | POINTS | 3.1 [3.0, 3.3] ABOVE | - | 51.8 [50.1, 53.5] | 0.67896 | 0.49241 |
| label:NER.spacy | found | 0.2692 | 9.5 [2.1, 20.0] | 36.54 [7.65, 82.91] | POINTS | 8.0 [1.0, 17.9] ABOVE | 0.3 [0.1, 0.6] | 75.0 [0.0, 100.0] | 1.0 | 0.75 |
| label:NER.spacy | top1 | 0.1666 | 0.4 [0.1, 0.7] | 2.19 [0.73, 3.93] | NO_LIFT | -5.9 [-6.3, -5.5] BELOW | - | 16.7 [0.0, 50.0] | 0.66667 | 0.0 |
| label:K.knn | found | 3.0897 | 6.8 [3.2, 11.4] | 26.27 [11.98, 48.06] | POINTS | 5.4 [2.5, 9.0] ABOVE | 1.7 [0.9, 3.0] | 68.2 [40.0, 87.5] | 0.90909 | 0.59091 |
| label:K.knn | top1 | 3.1629 | 0.2 [0.1, 0.3] | 1.21 [0.88, 1.60] | NO_LIFT | -6.1 [-6.4, -5.8] BELOW | - | 54.9 [41.7, 69.2] | 0.7451 | 0.54902 |
| group:S | found | 12.1346 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -1.5 [-2.5, -0.7] BELOW | 0.0 [0.0, 0.0] | - | None | None |
| group:S | top1 | 6.3867 | 9.4 [9.0, 9.8] | 56.00 [52.73, 59.56] | POINTS | 3.1 [3.0, 3.3] ABOVE | - | 51.8 [50.1, 53.5] | 0.67896 | 0.49241 |
| group:R | found | 0.2692 | 9.5 [2.1, 20.0] | 36.54 [7.65, 82.91] | POINTS | 8.0 [1.0, 17.9] ABOVE | 0.3 [0.1, 0.6] | 75.0 [0.0, 100.0] | 1.0 | 0.75 |
| group:R | top1 | 0.1666 | 0.4 [0.1, 0.7] | 2.19 [0.73, 3.93] | NO_LIFT | -5.9 [-6.3, -5.5] BELOW | - | 16.7 [0.0, 50.0] | 0.66667 | 0.0 |
| group:K | found | 3.0897 | 6.8 [3.2, 11.4] | 26.27 [11.98, 48.06] | POINTS | 5.4 [2.5, 9.0] ABOVE | 1.7 [0.9, 3.0] | 68.2 [40.0, 87.5] | 0.90909 | 0.59091 |
| group:K | top1 | 3.1629 | 0.2 [0.1, 0.3] | 1.21 [0.88, 1.60] | NO_LIFT | -6.1 [-6.4, -5.8] BELOW | - | 54.9 [41.7, 69.2] | 0.7451 | 0.54902 |
| any | found | 15.1987 | 1.5 [0.7, 2.5] | 5.66 [2.58, 10.20] | POINTS | - | 1.9 [0.9, 3.3] | 60.9 [30.8, 81.8] | 0.86957 | 0.52174 |
| any | top1 | 9.6732 | 6.3 [6.0, 6.6] | 37.38 [35.25, 39.65] | POINTS | - | - | 8.5 [7.6, 9.4] | 0.19872 | 0.11955 |

Multi-label minus single-label precision (found): 3.2 [-1.5, 10.6] LEVEL (43 / 2328 pairs).

Multi-label minus single-label precision (top1): -5.1 [-6.1, -4.0] BELOW (417 / 94235 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | slot | 79.1 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| found | K.knn | 18.7 | 7.0 [3.3, 11.8] | 26.79 [12.08, 48.71] |
| found | NER.spacy + K.knn | 1.1 | 8.0 [0.0, 19.2] | 30.70 [0.00, 79.69] |
| found | slot + K.knn | 0.4 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| top1 | slot | 65.9 | 9.4 [9.0, 9.8] | 56.07 [52.79, 59.61] |
| top1 | K.knn | 32.3 | 0.2 [0.1, 0.3] | 1.17 [0.84, 1.55] |
| top1 | NER.spacy | 1.4 | 0.2 [0.0, 0.4] | 0.92 [0.00, 2.30] |

## REC webqsp (1503 questions with a gold)

| unit | kind | pairs per anchor | precision | lift | call | vs any | reach | rank1 | top2 | rank1 by cos |
|---|---|---|---|---|---|---|---|---|---|---|
| label:slot | found | 158.6882 | 0.2 [0.0, 0.7] | 1.61 [0.20, 4.57] | NO_LIFT | -0.2 [-0.3, -0.1] BELOW | 6.5 [0.9, 17.6] | 23.1 [0.0, 50.0] | 0.23077 | 0.23077 |
| label:slot | top1 | 58.0998 | 1.8 [1.3, 2.4] | 15.34 [11.40, 20.08] | POINTS | 0.1 [0.1, 0.2] ABOVE | - | 18.7 [15.5, 22.1] | 0.35689 | 0.24028 |
| label:NER.spacy | found | 2.3441 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -0.4 [-0.9, -0.1] BELOW | 0.0 [0.0, 0.0] | - | None | None |
| label:NER.spacy | top1 | 2.6587 | 2.4 [2.0, 2.9] | 20.53 [16.42, 25.22] | POINTS | 0.7 [0.1, 1.4] ABOVE | - | 44.7 [34.4, 54.1] | 0.55319 | 0.39362 |
| label:K.knn | found | 4.0484 | 6.8 [4.1, 9.6] | 44.75 [27.65, 68.69] | POINTS | 6.4 [3.7, 9.1] ABOVE | 4.0 [2.5, 6.0] | 71.4 [54.5, 86.4] | 0.91429 | 0.68571 |
| label:K.knn | top1 | 5.5682 | 0.9 [0.7, 1.2] | 7.97 [5.98, 10.40] | POINTS | -0.7 [-1.3, -0.3] BELOW | - | 36.9 [25.4, 48.6] | 0.6 | 0.43077 |
| group:S | found | 158.6882 | 0.2 [0.0, 0.7] | 1.61 [0.20, 4.57] | NO_LIFT | -0.2 [-0.3, -0.1] BELOW | 6.5 [0.9, 17.6] | 23.1 [0.0, 50.0] | 0.23077 | 0.23077 |
| group:S | top1 | 58.0998 | 1.8 [1.3, 2.4] | 15.34 [11.40, 20.08] | POINTS | 0.1 [0.1, 0.2] ABOVE | - | 18.7 [15.5, 22.1] | 0.35689 | 0.24028 |
| group:R | found | 2.3441 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] | BELOW_BASE | -0.4 [-0.9, -0.1] BELOW | 0.0 [0.0, 0.0] | - | None | None |
| group:R | top1 | 2.6587 | 2.4 [2.0, 2.9] | 20.53 [16.42, 25.22] | POINTS | 0.7 [0.1, 1.4] ABOVE | - | 44.7 [34.4, 54.1] | 0.55319 | 0.39362 |
| group:K | found | 4.0484 | 6.8 [4.1, 9.6] | 44.75 [27.65, 68.69] | POINTS | 6.4 [3.7, 9.1] ABOVE | 4.0 [2.5, 6.0] | 71.4 [54.5, 86.4] | 0.91429 | 0.68571 |
| group:K | top1 | 5.5682 | 0.9 [0.7, 1.2] | 7.97 [5.98, 10.40] | POINTS | -0.7 [-1.3, -0.3] BELOW | - | 36.9 [25.4, 48.6] | 0.6 | 0.43077 |
| any | found | 162.5054 | 0.4 [0.1, 0.9] | 2.65 [1.02, 5.75] | POINTS | - | 10.3 [4.2, 21.2] | 41.3 [24.2, 59.1] | 0.52174 | 0.45652 |
| any | top1 | 64.2914 | 1.7 [1.3, 2.2] | 14.26 [10.77, 18.50] | POINTS | - | - | 6.3 [4.5, 8.2] | 0.12086 | 0.10265 |

Multi-label minus single-label precision (found): 0.0 [-0.8, 1.0] LEVEL (474 / 29752 pairs).

Multi-label minus single-label precision (top1): 2.6 [1.7, 3.5] ABOVE (2989 / 93641 pairs).

| kind | labels | share of pairs | precision | lift |
|---|---|---|---|---|
| found | slot | 96.1 | 0.2 [0.0, 0.7] | 1.59 [0.19, 4.57] |
| found | K.knn | 2.0 | 8.2 [4.9, 11.7] | 54.33 [33.88, 82.78] |
| found | slot + NER.spacy | 1.0 | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| top1 | slot | 87.5 | 1.7 [1.3, 2.3] | 14.58 [10.64, 19.37] |
| top1 | K.knn | 7.3 | 0.6 [0.4, 0.8] | 4.96 [3.26, 6.99] |
| top1 | NER.spacy | 2.1 | 0.2 [0.0, 0.4] | 1.67 [0.38, 3.58] |
| top1 | slot + NER.spacy | 1.7 | 5.2 [4.1, 6.6] | 44.80 [34.27, 57.62] |
| top1 | slot + K.knn | 1.1 | 3.2 [2.0, 4.5] | 27.17 [17.30, 39.04] |

## Label weights named for a later refit (ln of top1 lift; not adopted)

```
{
 "D83E": {
  "squad": {
   "R.mention": 0.0956,
   "R.name_comention": 0.8815,
   "K.knn": 0.9568
  },
  "hotpotqa": {
   "R.mention": 3.2671,
   "R.name_comention": 2.6031,
   "R.title_comention": 4.4523,
   "K.knn": 3.0409
  },
  "2wiki": {
   "R.mention": 3.5465,
   "R.name_comention": 2.7785,
   "R.title_comention": 4.6273,
   "K.knn": 3.4821
  },
  "musique": {
   "R.mention": 2.9885,
   "R.name_comention": 4.1043,
   "R.title_comention": 5.4797,
   "K.knn": 4.6944
  },
  "metaqa": {
   "S": 4.0254,
   "R.name_comention": -0.5044,
   "K.knn": 0.1937
  },
  "webqsp": {
   "S": 2.7303,
   "R.name_comention": 2.2633,
   "K.knn": 2.0754
  }
 },
 "B1V": {
  "musique": {
   "R.mention": 4.1167,
   "R.name_comention": 4.2146,
   "R.title_comention": 4.7334,
   "K.knn": 4.8453
  },
  "hotpotqa": {
   "R.mention": 3.7494,
   "R.name_comention": 2.0832,
   "R.title_comention": 3.7959,
   "K.knn": 2.4607
  },
  "2wiki": {
   "R.mention": 4.4534,
   "R.name_comention": 2.5321,
   "R.title_comention": 4.393,
   "K.knn": 2.6699
  }
 }
}
```
