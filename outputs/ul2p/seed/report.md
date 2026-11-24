# UL2S report: the seed-anchored label ledger on the e3v layout

Declared in docs/UL2S_SEED_ANCHORED_LABEL_LEDGER.md. Written 2026-11-24T23:40:54Z. Decides nothing.

## Expectations (rrf list, seed kind)

| id | call | reads | detail |
|---|---|---|---|
| E1 | HELD | 2 of 2 |  |
| E2 | HELD | 2 of 2 | multihoprag/RK: multi minus single ABOVE: held; squad/RK: multi minus single ABOVE: held |
| E3 | NOT_HELD | 8 of 14 | squad/RK: H5 vs any LEVEL; squad/RK: HM vs any ABOVE; squad/RK: HK1 vs any ABOVE; squad/RK: NT1 vs any LEVEL; squad/RK: HK2 vs any LEVEL; squad/RK: H10 vs any LEVEL |
| E4 | NOT_READ | 0 of 0 |  |
| E5 | NOT_READ | 0 of 0 |  |
| E6 | NOT_HELD | 0 of 1 | multihoprag/RK: found 0.00797 [0.00739, 0.00855], top1 0.01159 [0.01084, 0.01241]: not |
| E7 | HELD | 1 of 1 | multihoprag/RK: GOLD_LINKED 0.0: held |

## What follows (section 7: names only; no rule change, no carry)

- Candidate label weights (POINTS and ABOVE the any-link unit): multihoprag/RK: label:R.name_comention; multihoprag/RK: label:K.knn; squad/RK: label:R.name_comention; squad/RK: label:K.knn
- Per label, across the arms where it has enough targets: label:K.knn: ABOVE and POINTS on 2 of 2 arms with at least 20 targets (UNIVERSAL); label:R.mention: ABOVE and POINTS on 0 of 1 arms with at least 20 targets; label:R.name_comention: ABOVE and POINTS on 2 of 2 arms with at least 20 targets (UNIVERSAL)
- Candidate tighter cuts (a form BELOW the any-link unit whose first quintile is ABOVE it): multihoprag/RK: HM
- Cost without precision (a label holding at least 5% of the any-link candidates, BELOW or without lift): none
- The agreement candidate (multi minus single ABOVE): 2 of 2 arms ({'multihoprag/RK': 'ABOVE', 'squad/RK': 'ABOVE'})
- The rules' gap (shares of the missed golds, rrf seeds):

| arm | no rule reaches (ISOLATED + LINKED_ELSE) % | ISOLATED % | LINKED_ELSE % | GOLD_LINKED % | SEED_B0 % | SEED_FORM % |
|---|---|---|---|---|---|---|
| multihoprag/RK | 0.40 | 0.00 | 0.40 | 0.00 | 97.83 | 1.77 |
| squad/RK | 38.21 | 0.00 | 38.21 | n/a | 51.83 | 9.96 |

## multihoprag


### multihoprag / RK / s1eval / rrf list: 1,130 questions, 2,985 golds, 1,741 found, 1,244 missed (783 questions)


**seed anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0108 | 3.28 [0.00, 8.51] | 0.136 | NO_LIFT | 0.16 [0.00, 0.41] | 1.0 | 2.18 [-1.13, 7.40] LEVEL |
| label:R.name_comention | 25,005 | 83.9207 | 1.11 [1.04, 1.17] | 0.178 | POINTS | 97.51 [96.50, 98.41] | 20.6 | 0.00 [0.00, 0.01] ABOVE |
| label:R.title_comention | 75 | 0.214 | 0.83 [0.34, 1.37] | 0.172 | POINTS | 0.80 [0.33, 1.34] | 7.5 | -0.27 [-0.75, 0.27] LEVEL |
| label:K.knn | 1,366 | 3.6368 | 6.52 [6.04, 7.06] | 0.190 | POINTS | 55.55 [53.06, 58.02] | 2.0 | 5.42 [4.96, 5.92] ABOVE |
| group:R | 25,080 | 84.1347 | 1.10 [1.04, 1.17] | 0.178 | POINTS | 97.51 [96.50, 98.41] | 20.7 | 0.00 [0.00, 0.01] ABOVE |
| group:K | 263 | 0.5966 | 0.62 [0.33, 0.94] | 0.169 | POINTS | 1.37 [0.75, 2.04] | 15.5 | -0.48 [-0.78, -0.14] BELOW |
| any | 25,343 | 84.7313 | 1.10 [1.04, 1.17] | 0.178 | POINTS | 97.83 [96.91, 98.66] | 20.8 | - |
| form:H5 | 2,387 | 9.6274 | 0.44 [0.37, 0.51] | 0.182 | POINTS | 14.47 [12.34, 16.73] | 13.3 | -0.67 [-0.77, -0.57] BELOW |
| form:HM | 1,332 | 5.1186 | 0.68 [0.57, 0.80] | 0.179 | POINTS | 12.06 [10.13, 14.03] | 8.9 | -0.42 [-0.56, -0.28] BELOW |
| form:HK1 | 2,712 | 10.7211 | 0.38 [0.32, 0.44] | 0.181 | POINTS | 13.83 [11.77, 16.04] | 15.8 | -0.72 [-0.81, -0.63] BELOW |
| form:NTc | 13,668 | 40.8586 | 0.11 [0.10, 0.13] | 0.173 | BELOW_BASE | 16.48 [14.21, 18.73] | 66.7 | -0.99 [-1.06, -0.92] BELOW |
| form:NT1 | 4,604 | 15.6265 | 0.19 [0.15, 0.22] | 0.174 | NO_LIFT | 10.21 [8.38, 11.96] | 36.3 | -0.92 [-0.99, -0.84] BELOW |
| form:HK2 | 4,886 | 19.8273 | 0.29 [0.24, 0.33] | 0.183 | POINTS | 18.57 [16.08, 21.15] | 21.2 | -0.82 [-0.90, -0.74] BELOW |
| form:H10 | 4,588 | 18.4393 | 0.36 [0.31, 0.41] | 0.183 | POINTS | 19.13 [16.68, 21.69] | 19.3 | -0.75 [-0.83, -0.66] BELOW |
| all | 46,761 | 155.1874 | 0.67 [0.63, 0.70] | 0.177 | POINTS | 99.60 [99.20, 99.92] | 37.7 | -0.44 [-0.47, -0.40] BELOW |

**top1 anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0097 | 0.00 [0.00, 0.00] | 0.075 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -1.16 [-1.24, -1.08] BELOW |
| label:R.name_comention | 25,005 | 79.4611 | 1.17 [1.09, 1.25] | 0.174 | POINTS | 84.16 [81.84, 86.44] | 23.9 | 0.01 [0.00, 0.01] ABOVE |
| label:R.title_comention | 75 | 0.1858 | 0.48 [0.00, 1.55] | 0.149 | NO_LIFT | 0.08 [0.00, 0.25] | 75.0 | -0.68 [-1.22, 0.40] LEVEL |
| label:K.knn | 1,366 | 2.8788 | 8.36 [7.40, 9.37] | 0.206 | POINTS | 21.87 [19.61, 24.18] | 5.0 | 7.20 [6.27, 8.17] ABOVE |
| group:R | 25,080 | 79.6469 | 1.16 [1.09, 1.25] | 0.174 | POINTS | 84.24 [81.89, 86.52] | 23.9 | 0.01 [0.00, 0.01] ABOVE |
| group:K | 263 | 0.546 | 0.32 [0.00, 0.85] | 0.202 | NO_LIFT | 0.16 [0.00, 0.42] | 131.5 | -0.83 [-1.19, -0.29] BELOW |
| any | 25,343 | 80.1929 | 1.16 [1.08, 1.24] | 0.174 | POINTS | 84.41 [82.11, 86.65] | 24.1 | - |
| all | 46,761 | 147.0681 | 0.70 [0.66, 0.74] | 0.176 | POINTS | 93.81 [92.23, 95.25] | 40.1 | -0.46 [-0.50, -0.42] BELOW |

**found anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0172 | 0.00 [0.00, 0.00] | 0.110 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -0.80 [-0.85, -0.74] BELOW |
| label:R.name_comention | 25,005 | 80.5681 | 0.80 [0.74, 0.86] | 0.128 | POINTS | 85.66 [83.19, 88.09] | 29.7 | 0.00 [-0.00, 0.01] LEVEL |
| label:R.title_comention | 75 | 0.2366 | 1.94 [0.73, 3.47] | 0.123 | POINTS | 0.81 [0.31, 1.44] | 9.4 | 1.15 [-0.05, 2.69] LEVEL |
| label:K.knn | 1,366 | 2.9104 | 5.35 [4.66, 6.04] | 0.140 | POINTS | 25.23 [22.55, 27.94] | 5.5 | 4.55 [3.90, 5.23] ABOVE |
| group:R | 25,080 | 80.8047 | 0.80 [0.74, 0.86] | 0.128 | POINTS | 86.17 [83.72, 88.60] | 29.6 | 0.00 [0.00, 0.01] ABOVE |
| group:K | 263 | 0.5307 | 0.22 [0.00, 0.56] | 0.102 | NO_LIFT | 0.20 [0.00, 0.52] | 131.5 | -0.58 [-0.82, -0.23] BELOW |
| any | 25,343 | 81.3354 | 0.80 [0.74, 0.85] | 0.128 | POINTS | 86.27 [83.82, 88.67] | 29.9 | - |
| all | 46,761 | 149.7209 | 0.50 [0.47, 0.53] | 0.126 | POINTS | 95.22 [93.70, 96.67] | 50.0 | -0.30 [-0.33, -0.27] BELOW |

Classes of the missed golds: SEED_B0 97.83 [96.90, 98.64] (1,217); SEED_FORM 1.77 [1.10, 2.55] (22); GOLD_LINKED 0.00 [0.00, 0.00] (0); LINKED_ELSE 0.40 [0.08, 0.80] (5); ISOLATED 0.00 [0.00, 0.00] (0)

Multi minus single: 6.81 [6.25, 7.40] ABOVE.

Forms, first and last rank quintile (precision %; first against last):

- H5: first 1.09 [0.88, 1.31] (139 targets), last 0.03 [0.00, 0.07] (2 targets), ABOVE
- HM: first 1.83 [1.44, 2.24] (115 targets), last 0.17 [0.03, 0.36] (5 targets), ABOVE
- HK1: first 0.75 [0.59, 0.90] (108 targets), last 0.01 [0.00, 0.03] (1 targets), ABOVE
- NTc: first 0.09 [0.06, 0.12] (37 targets), last 0.06 [0.04, 0.09] (34 targets), LEVEL
- NT1: first 0.49 [0.39, 0.59] (110 targets), last 0.01 [0.00, 0.03] (1 targets), ABOVE
- HK2: first 0.83 [0.70, 0.96] (234 targets), last 0.02 [0.00, 0.04] (3 targets), ABOVE
- H10: first 0.92 [0.76, 1.08] (240 targets), last 0.02 [0.00, 0.05] (3 targets), ABOVE

Cut frontier (exact label sets, descending precision, cumulative):

| labels | candidates | targets | precision % | cum candidates | cum precision % | cum reach % |
|---|---|---|---|---|---|---|
| R.name_comention+K.knn | 17,169 | 1,319 | 7.683 | 17,169 | 7.683 | 54.58 |
| R.mention+R.name_comention | 61 | 2 | 3.279 | 17,230 | 7.667 | 54.74 |
| R.name_comention | 456,922 | 3,922 | 0.858 | 474,152 | 1.106 | 97.51 |
| R.title_comention | 1,201 | 10 | 0.833 | 475,353 | 1.105 | 97.51 |
| K.knn | 3,371 | 21 | 0.623 | 478,724 | 1.102 | 97.83 |
| R.title_comention+K.knn | 8 | 0 | 0.000 | 478,732 | 1.102 | 97.83 |

### multihoprag / RK / s1eval / dense list: 1,130 questions, 2,985 golds, 1,828 found, 1,157 missed (735 questions)


**seed anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0115 | 3.08 [0.00, 7.94] | 0.17 [0.00, 0.44] | 1.0 |
| label:R.name_comention | 25,005 | 83.1419 | 1.05 [0.98, 1.12] | 96.97 [95.99, 97.93] | 22.3 |
| label:R.title_comention | 75 | 0.2112 | 1.09 [0.52, 1.71] | 1.12 [0.53, 1.77] | 5.8 |
| label:K.knn | 1,366 | 3.6044 | 5.32 [4.87, 5.79] | 54.71 [52.03, 57.28] | 2.2 |
| group:R | 25,080 | 83.3531 | 1.05 [0.98, 1.12] | 96.97 [95.99, 97.93] | 22.4 |
| group:K | 263 | 0.6189 | 0.52 [0.22, 0.89] | 1.12 [0.52, 1.84] | 20.2 |
| any | 25,343 | 83.972 | 1.05 [0.98, 1.11] | 97.41 [96.48, 98.28] | 22.5 |
| all | 46,761 | 155.5926 | 0.62 [0.58, 0.66] | 99.22 [98.69, 99.66] | 40.7 |

**top1 anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0115 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| label:R.name_comention | 25,005 | 77.9248 | 1.13 [1.06, 1.21] | 86.08 [83.84, 88.27] | 25.1 |
| label:R.title_comention | 75 | 0.215 | 0.82 [0.00, 2.17] | 0.17 [0.00, 0.44] | 37.5 |
| label:K.knn | 1,366 | 3.0018 | 6.66 [5.81, 7.56] | 19.53 [17.27, 21.84] | 6.0 |
| group:R | 25,080 | 78.1398 | 1.13 [1.05, 1.21] | 86.26 [84.10, 88.47] | 25.1 |
| group:K | 263 | 0.5788 | 0.31 [0.00, 0.77] | 0.17 [0.00, 0.44] | 131.5 |
| any | 25,343 | 78.7186 | 1.12 [1.05, 1.20] | 86.43 [84.25, 88.62] | 25.3 |
| all | 46,761 | 143.8319 | 0.67 [0.63, 0.71] | 93.95 [92.47, 95.43] | 43.0 |

**found anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 2 | 0.017 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| label:R.name_comention | 25,005 | 80.9327 | 0.76 [0.70, 0.82] | 86.67 [84.32, 89.03] | 30.1 |
| label:R.title_comention | 75 | 0.2188 | 1.75 [0.55, 3.16] | 0.73 [0.22, 1.28] | 10.7 |
| label:K.knn | 1,366 | 2.9617 | 4.38 [3.77, 5.02] | 22.29 [19.65, 24.95] | 6.4 |
| group:R | 25,080 | 81.1515 | 0.76 [0.70, 0.82] | 86.98 [84.66, 89.29] | 30.0 |
| group:K | 263 | 0.535 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| any | 25,343 | 81.6865 | 0.75 [0.70, 0.82] | 86.98 [84.66, 89.29] | 30.4 |
| all | 46,761 | 150.599 | 0.46 [0.43, 0.50] | 94.69 [93.14, 96.12] | 51.4 |

Classes of the missed golds: SEED_B0 97.41 [96.41, 98.25] (1,127); SEED_FORM 1.81 [1.11, 2.66] (21); GOLD_LINKED 0.00 [0.00, 0.00] (0); LINKED_ELSE 0.78 [0.34, 1.31] (9); ISOLATED 0.00 [0.00, 0.00] (0)


### multihoprag / RK / s1sel / rrf list: 224 questions, 575 golds, 353 found, 222 missed (146 questions)


**seed anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0107 | 0.00 [0.00, 0.00] | 0.138 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -0.98 [-1.13, -0.85] BELOW |
| label:R.name_comention | 25,005 | 84.0554 | 0.99 [0.85, 1.14] | 0.160 | POINTS | 96.40 [93.75, 98.64] | 116.8 | 0.01 [0.00, 0.01] LEVEL |
| label:R.title_comention | 75 | 0.2357 | 0.76 [0.00, 2.00] | 0.160 | NO_LIFT | 0.90 [0.00, 2.37] | 37.5 | -0.22 [-1.05, 1.05] LEVEL |
| label:K.knn | 1,366 | 3.6268 | 5.37 [4.37, 6.45] | 0.174 | POINTS | 54.95 [48.94, 61.09] | 11.2 | 4.39 [3.49, 5.38] ABOVE |
| group:R | 25,080 | 84.2911 | 0.98 [0.85, 1.14] | 0.160 | POINTS | 96.40 [93.75, 98.64] | 117.2 | 0.00 [0.00, 0.01] ABOVE |
| group:K | 263 | 0.5527 | 0.32 [0.00, 0.83] | 0.149 | NO_LIFT | 0.90 [0.00, 2.38] | 131.5 | -0.66 [-1.04, -0.12] BELOW |
| any | 25,343 | 84.8438 | 0.98 [0.85, 1.13] | 0.160 | POINTS | 96.85 [94.37, 99.03] | 117.9 | - |
| form:H5 | 2,387 | 9.3143 | 0.48 [0.31, 0.66] | 0.168 | POINTS | 16.22 [11.16, 21.62] | 66.3 | -0.50 [-0.73, -0.28] BELOW |
| form:HM | 1,332 | 5.0723 | 0.70 [0.44, 0.99] | 0.164 | POINTS | 13.51 [8.84, 18.43] | 44.4 | -0.28 [-0.59, 0.04] LEVEL |
| form:HK1 | 2,712 | 10.3812 | 0.29 [0.19, 0.41] | 0.167 | POINTS | 13.06 [8.50, 18.10] | 93.5 | -0.69 [-0.87, -0.51] BELOW |
| form:NTc | 13,668 | 41.3429 | 0.10 [0.07, 0.14] | 0.158 | BELOW_BASE | 17.12 [11.76, 22.62] | 359.7 | -0.88 [-1.04, -0.74] BELOW |
| form:NT1 | 4,604 | 15.4571 | 0.13 [0.07, 0.19] | 0.158 | NO_LIFT | 9.01 [5.42, 13.05] | 230.2 | -0.85 [-1.01, -0.70] BELOW |
| form:HK2 | 4,886 | 19.3027 | 0.25 [0.18, 0.34] | 0.169 | POINTS | 19.37 [13.83, 25.24] | 113.6 | -0.73 [-0.90, -0.57] BELOW |
| form:H10 | 4,588 | 18.0125 | 0.36 [0.25, 0.47] | 0.169 | POINTS | 21.17 [15.12, 27.78] | 97.6 | -0.62 [-0.82, -0.45] BELOW |
| all | 46,761 | 155.0429 | 0.59 [0.51, 0.67] | 0.161 | POINTS | 98.65 [96.90, 100.00] | 213.5 | -0.39 [-0.46, -0.32] BELOW |

**top1 anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0 | - | - | - | 0.00 [0.00, 0.00] | - | - |
| label:R.name_comention | 25,005 | 77.0446 | 1.07 [0.91, 1.24] | 0.157 | POINTS | 82.88 [76.76, 88.52] | 135.9 | -0.00 [-0.02, 0.01] LEVEL |
| label:R.title_comention | 75 | 0.2366 | 1.89 [0.00, 6.78] | 0.131 | NO_LIFT | 0.45 [0.00, 1.44] | 75.0 | 0.82 [-1.20, 5.62] LEVEL |
| label:K.knn | 1,366 | 2.8795 | 5.43 [3.86, 7.23] | 0.181 | POINTS | 15.77 [11.64, 20.38] | 39.0 | 4.36 [2.83, 6.08] ABOVE |
| group:R | 25,080 | 77.2812 | 1.07 [0.91, 1.25] | 0.157 | POINTS | 83.33 [77.29, 89.00] | 135.6 | 0.00 [-0.01, 0.01] LEVEL |
| group:K | 263 | 0.5268 | 0.85 [0.00, 2.83] | 0.171 | NO_LIFT | 0.45 [0.00, 1.44] | 263.0 | -0.22 [-1.19, 1.75] LEVEL |
| any | 25,343 | 77.808 | 1.07 [0.91, 1.24] | 0.157 | POINTS | 83.78 [77.92, 89.43] | 136.3 | - |
| all | 46,761 | 144.1964 | 0.65 [0.56, 0.74] | 0.159 | POINTS | 94.59 [90.60, 97.92] | 222.7 | -0.42 [-0.51, -0.33] BELOW |

**found anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0113 | 0.00 [0.00, 0.00] | 0.124 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -0.72 [-0.85, -0.60] BELOW |
| label:R.name_comention | 25,005 | 79.2408 | 0.73 [0.60, 0.86] | 0.115 | POINTS | 86.29 [80.44, 91.81] | 165.6 | 0.00 [-0.01, 0.01] LEVEL |
| label:R.title_comention | 75 | 0.2408 | 1.18 [0.00, 4.17] | 0.099 | NO_LIFT | 0.57 [0.00, 1.84] | 75.0 | 0.45 [-0.81, 3.44] LEVEL |
| label:K.knn | 1,366 | 2.9348 | 3.76 [2.63, 5.02] | 0.120 | POINTS | 20.00 [14.59, 26.19] | 39.0 | 3.04 [1.97, 4.23] ABOVE |
| group:R | 25,080 | 79.4816 | 0.73 [0.61, 0.86] | 0.115 | POINTS | 86.29 [80.44, 91.81] | 166.1 | 0.01 [0.00, 0.01] ABOVE |
| group:K | 263 | 0.5071 | 0.00 [0.00, 0.00] | 0.092 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -0.72 [-0.85, -0.60] BELOW |
| any | 25,343 | 79.9887 | 0.72 [0.60, 0.85] | 0.115 | POINTS | 86.29 [80.44, 91.81] | 167.8 | - |
| all | 46,761 | 149.0963 | 0.45 [0.38, 0.52] | 0.114 | POINTS | 94.86 [91.46, 97.86] | 281.7 | -0.27 [-0.34, -0.21] BELOW |

Classes of the missed golds: SEED_B0 96.85 [94.30, 98.76] (215); SEED_FORM 1.80 [0.44, 3.65] (4); GOLD_LINKED 0.00 [0.00, 0.00] (0); LINKED_ELSE 1.35 [0.00, 3.18] (3); ISOLATED 0.00 [0.00, 0.00] (0)

Multi minus single: 5.47 [4.38, 6.66] ABOVE.

Forms, first and last rank quintile (precision %; first against last):

- H5: first 1.26 [0.73, 1.84] (31 targets), last 0.27 [0.00, 0.65] (4 targets), ABOVE
- HM: first 2.31 [1.32, 3.40] (28 targets), last 0.17 [0.00, 0.56] (1 targets), ABOVE
- HK1: first 0.60 [0.27, 0.97] (17 targets), last 0.06 [0.00, 0.17] (1 targets), ABOVE
- NTc: first 0.07 [0.01, 0.15] (6 targets), last 0.02 [0.00, 0.05] (2 targets), LEVEL
- NT1: first 0.37 [0.18, 0.60] (16 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE
- HK2: first 0.68 [0.43, 0.96] (37 targets), last 0.03 [0.00, 0.10] (1 targets), ABOVE
- H10: first 0.92 [0.56, 1.31] (46 targets), last 0.10 [0.00, 0.22] (3 targets), ABOVE

Cut frontier (exact label sets, descending precision, cumulative):

| labels | candidates | targets | precision % | cum candidates | cum precision % | cum reach % |
|---|---|---|---|---|---|---|
| R.name_comention+K.knn | 3,443 | 216 | 6.274 | 3,443 | 6.274 | 54.05 |
| R.name_comention | 90,687 | 712 | 0.785 | 94,130 | 0.986 | 96.40 |
| R.title_comention | 264 | 2 | 0.758 | 94,394 | 0.985 | 96.40 |
| K.knn | 619 | 2 | 0.323 | 95,013 | 0.981 | 96.85 |
| R.mention+R.name_comention | 12 | 0 | 0.000 | 95,025 | 0.981 | 96.85 |

### multihoprag / RK / s1sel / dense list: 224 questions, 575 golds, 353 found, 222 missed (148 questions)


**seed anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0089 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| label:R.name_comention | 25,005 | 82.3375 | 1.00 [0.84, 1.14] | 95.49 [92.38, 98.24] | 117.9 |
| label:R.title_comention | 75 | 0.1973 | 0.91 [0.00, 2.22] | 0.90 [0.00, 2.34] | 37.5 |
| label:K.knn | 1,366 | 3.5018 | 5.02 [4.04, 6.07] | 52.25 [45.91, 58.65] | 11.8 |
| group:R | 25,080 | 82.5348 | 1.00 [0.84, 1.14] | 95.49 [92.38, 98.24] | 118.3 |
| group:K | 263 | 0.5964 | 0.60 [0.14, 1.24] | 1.80 [0.43, 3.69] | 65.8 |
| any | 25,343 | 83.1312 | 0.99 [0.84, 1.14] | 96.40 [93.45, 98.97] | 118.4 |
| all | 46,761 | 153.633 | 0.59 [0.51, 0.67] | 99.10 [97.72, 100.00] | 212.6 |

**top1 anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0134 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| label:R.name_comention | 25,005 | 79.5938 | 1.05 [0.90, 1.22] | 84.69 [79.45, 89.24] | 133.0 |
| label:R.title_comention | 75 | 0.2411 | 1.85 [0.00, 6.38] | 0.45 [0.00, 1.45] | 75.0 |
| label:K.knn | 1,366 | 3.0938 | 5.20 [3.60, 7.00] | 16.22 [11.48, 21.59] | 37.9 |
| group:R | 25,080 | 79.8348 | 1.06 [0.90, 1.22] | 85.14 [79.83, 89.71] | 132.7 |
| group:K | 263 | 0.5804 | 0.77 [0.00, 2.52] | 0.45 [0.00, 1.44] | 263.0 |
| any | 25,343 | 80.4152 | 1.05 [0.90, 1.22] | 85.59 [80.66, 90.09] | 133.4 |
| all | 46,761 | 148.2857 | 0.64 [0.55, 0.73] | 95.05 [91.58, 97.88] | 221.6 |

**found anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0113 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| label:R.name_comention | 25,005 | 78.4674 | 0.77 [0.64, 0.90] | 87.29 [81.99, 91.96] | 158.3 |
| label:R.title_comention | 75 | 0.2068 | 1.37 [0.00, 4.84] | 0.55 [0.00, 1.80] | 75.0 |
| label:K.knn | 1,366 | 2.9122 | 4.09 [2.85, 5.38] | 22.65 [16.67, 28.90] | 33.3 |
| group:R | 25,080 | 78.6742 | 0.77 [0.64, 0.91] | 87.29 [81.99, 91.96] | 158.7 |
| group:K | 263 | 0.5184 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| any | 25,343 | 79.1926 | 0.76 [0.64, 0.90] | 87.29 [81.99, 91.96] | 160.4 |
| all | 46,761 | 147.3598 | 0.46 [0.39, 0.54] | 95.03 [91.76, 97.89] | 271.9 |

Classes of the missed golds: SEED_B0 96.40 [93.46, 98.70] (214); SEED_FORM 2.70 [0.89, 5.09] (6); GOLD_LINKED 0.45 [0.00, 1.42] (1); LINKED_ELSE 0.45 [0.00, 1.40] (1); ISOLATED 0.00 [0.00, 0.00] (0)


## squad


### squad / RK / s1eval / rrf list: 11,873 questions, 11,873 golds, 10,478 found, 1,395 missed (1,395 questions)


**seed anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 878,200 | 71.9512 | 0.04 [0.04, 0.04] | 0.001 | POINTS | 42.72 [40.21, 45.32] | 1473.5 | -0.00 [-0.00, -0.00] LEVEL |
| label:R.name_comention | 173,645 | 17.1051 | 0.07 [0.07, 0.08] | 0.001 | POINTS | 25.52 [23.24, 27.79] | 487.8 | 0.03 [0.02, 0.04] ABOVE |
| label:R.title_comention | 2,809 | 0.2126 | 0.00 [0.00, 0.00] | 0.001 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -0.04 [-0.05, -0.04] BELOW |
| label:K.knn | 43,350 | 3.6239 | 0.21 [0.18, 0.23] | 0.001 | POINTS | 24.88 [22.70, 27.16] | 124.9 | 0.16 [0.14, 0.18] ABOVE |
| group:R | 989,371 | 83.4017 | 0.04 [0.04, 0.05] | 0.001 | POINTS | 49.46 [46.87, 52.07] | 1433.9 | -0.00 [-0.00, -0.00] BELOW |
| group:K | 6,498 | 0.585 | 0.18 [0.13, 0.22] | 0.001 | POINTS | 4.01 [2.99, 5.08] | 116.0 | 0.13 [0.09, 0.18] ABOVE |
| any | 995,869 | 83.9867 | 0.04 [0.04, 0.05] | 0.001 | POINTS | 51.83 [49.27, 54.43] | 1377.4 | - |
| form:H5 | 60,681 | 5.6894 | 0.05 [0.04, 0.06] | 0.001 | POINTS | 9.68 [8.16, 11.34] | 449.5 | 0.01 [-0.00, 0.02] LEVEL |
| form:HM | 34,187 | 3.352 | 0.07 [0.05, 0.08] | 0.001 | POINTS | 7.60 [6.25, 9.02] | 322.5 | 0.03 [0.01, 0.04] ABOVE |
| form:HK1 | 38,713 | 3.7596 | 0.06 [0.05, 0.07] | 0.001 | POINTS | 8.03 [6.61, 9.51] | 345.7 | 0.01 [0.00, 0.03] ABOVE |
| form:NTc | 72,853 | 7.1158 | 0.03 [0.02, 0.03] | 0.001 | POINTS | 5.95 [4.82, 7.22] | 877.7 | -0.02 [-0.02, -0.01] BELOW |
| form:NT1 | 55,061 | 5.4187 | 0.04 [0.03, 0.05] | 0.001 | POINTS | 7.38 [6.07, 8.78] | 534.6 | -0.01 [-0.01, 0.00] LEVEL |
| form:HK2 | 72,729 | 7.0994 | 0.05 [0.04, 0.06] | 0.001 | POINTS | 10.47 [8.90, 12.08] | 498.1 | 0.00 [-0.00, 0.01] LEVEL |
| form:H10 | 116,928 | 11.0188 | 0.04 [0.04, 0.05] | 0.001 | POINTS | 14.19 [12.45, 16.10] | 590.5 | -0.00 [-0.01, 0.01] LEVEL |
| all | 1,249,143 | 108.3495 | 0.04 [0.04, 0.04] | 0.001 | POINTS | 61.79 [59.27, 64.32] | 1449.1 | -0.00 [-0.00, -0.00] BELOW |

**top1 anchors**

| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |
|---|---|---|---|---|---|---|---|---|
| label:R.mention | 878,200 | 63.4189 | 0.05 [0.05, 0.06] | 0.001 | POINTS | 27.96 [25.65, 30.32] | 2251.8 | -0.00 [-0.00, 0.00] LEVEL |
| label:R.name_comention | 173,645 | 16.1737 | 0.10 [0.09, 0.11] | 0.001 | POINTS | 13.76 [12.04, 15.61] | 904.4 | 0.05 [0.03, 0.06] ABOVE |
| label:R.title_comention | 2,809 | 0.1568 | 0.00 [0.00, 0.00] | 0.001 | BELOW_BASE | 0.00 [0.00, 0.00] | - | -0.05 [-0.06, -0.05] BELOW |
| label:K.knn | 43,350 | 3.1121 | 0.32 [0.27, 0.39] | 0.001 | POINTS | 8.53 [7.11, 10.09] | 364.3 | 0.27 [0.21, 0.33] ABOVE |
| group:R | 989,371 | 74.2993 | 0.05 [0.05, 0.06] | 0.001 | POINTS | 33.48 [31.08, 35.98] | 2118.6 | -0.00 [-0.00, -0.00] LEVEL |
| group:K | 6,498 | 0.5506 | 0.15 [0.06, 0.26] | 0.001 | POINTS | 0.72 [0.29, 1.23] | 649.8 | 0.10 [0.01, 0.21] ABOVE |
| any | 995,869 | 74.8498 | 0.05 [0.05, 0.06] | 0.001 | POINTS | 34.19 [31.77, 36.71] | 2087.8 | - |
| all | 1,249,143 | 98.0739 | 0.05 [0.05, 0.06] | 0.001 | POINTS | 42.29 [39.78, 44.82] | 2117.2 | -0.00 [-0.01, -0.00] BELOW |

Classes of the missed golds: SEED_B0 51.83 [49.25, 54.48] (723); SEED_FORM 9.96 [8.46, 11.47] (139); GOLD_LINKED n/a (0); LINKED_ELSE 38.21 [35.63, 40.79] (533); ISOLATED 0.00 [0.00, 0.00] (0)

Multi minus single: 0.11 [0.10, 0.13] ABOVE.

Forms, first and last rank quintile (precision %; first against last):

- H5: first 0.11 [0.08, 0.14] (89 targets), last 0.01 [0.01, 0.03] (9 targets), ABOVE
- HM: first 0.12 [0.08, 0.16] (54 targets), last 0.03 [0.01, 0.05] (10 targets), ABOVE
- HK1: first 0.13 [0.09, 0.17] (65 targets), last 0.01 [0.00, 0.01] (2 targets), ABOVE
- NTc: first 0.04 [0.03, 0.06] (32 targets), last 0.02 [0.01, 0.03] (19 targets), LEVEL
- NT1: first 0.09 [0.06, 0.11] (63 targets), last 0.00 [0.00, 0.01] (2 targets), ABOVE
- HK2: first 0.10 [0.08, 0.13] (97 targets), last 0.01 [0.00, 0.01] (4 targets), ABOVE
- H10: first 0.08 [0.06, 0.10] (127 targets), last 0.01 [0.01, 0.02] (17 targets), ABOVE

Cut frontier (exact label sets, descending precision, cumulative):

| labels | candidates | targets | precision % | cum candidates | cum precision % | cum reach % |
|---|---|---|---|---|---|---|
| R.name_comention+K.knn | 10,170 | 28 | 0.275 | 10,170 | 0.275 | 1.86 |
| R.mention+R.name_comention+K.knn | 59,757 | 144 | 0.241 | 69,927 | 0.246 | 10.32 |
| R.mention+K.knn | 109,303 | 211 | 0.193 | 179,230 | 0.214 | 21.29 |
| K.knn | 34,726 | 61 | 0.176 | 213,956 | 0.207 | 24.88 |
| R.mention+R.name_comention | 276,545 | 303 | 0.110 | 490,501 | 0.152 | 31.61 |
| R.name_comention | 668,972 | 268 | 0.040 | 1,159,473 | 0.088 | 38.50 |
| R.mention | 3,813,777 | 1,123 | 0.029 | 4,973,250 | 0.043 | 51.83 |
| R.mention+R.title_comention | 10,832 | 0 | 0.000 | 4,984,082 | 0.043 | 51.83 |
| R.mention+R.title_comention+K.knn | 1,171 | 0 | 0.000 | 4,985,253 | 0.043 | 51.83 |
| R.title_comention | 613 | 0 | 0.000 | 4,985,866 | 0.043 | 51.83 |
| R.title_comention+K.knn | 3 | 0 | 0.000 | 4,985,869 | 0.043 | 51.83 |

### squad / RK / s1eval / dense list: 11,873 questions, 11,873 golds, 9,937 found, 1,936 missed (1,936 questions)


**seed anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 878,200 | 70.9155 | 0.06 [0.05, 0.06] | 41.89 [39.75, 44.00] | 1082.9 |
| label:R.name_comention | 173,645 | 16.8342 | 0.10 [0.08, 0.10] | 25.16 [23.23, 27.03] | 356.6 |
| label:R.title_comention | 2,809 | 0.2024 | 0.03 [0.00, 0.06] | 0.15 [0.00, 0.36] | 936.3 |
| label:K.knn | 43,350 | 3.5919 | 0.31 [0.28, 0.34] | 25.16 [23.19, 27.11] | 89.0 |
| group:R | 989,371 | 82.1402 | 0.06 [0.05, 0.06] | 49.17 [47.03, 51.35] | 1039.3 |
| group:K | 6,498 | 0.5821 | 0.26 [0.20, 0.32] | 3.82 [3.03, 4.71] | 87.8 |
| any | 995,869 | 82.7223 | 0.06 [0.06, 0.06] | 51.65 [49.40, 53.86] | 995.9 |
| all | 1,249,143 | 106.7834 | 0.06 [0.05, 0.06] | 62.14 [59.95, 64.24] | 1038.4 |

**top1 anchors (core)**

| unit | pairs | price per anchor | precision % | reach % | graph price |
|---|---|---|---|---|---|
| label:R.mention | 878,200 | 63.6851 | 0.07 [0.07, 0.08] | 27.53 [25.62, 29.49] | 1647.7 |
| label:R.name_comention | 173,645 | 16.003 | 0.13 [0.11, 0.15] | 12.65 [11.23, 14.18] | 708.8 |
| label:R.title_comention | 2,809 | 0.1698 | 0.05 [0.00, 0.16] | 0.05 [0.00, 0.16] | 2809.0 |
| label:K.knn | 43,350 | 3.0195 | 0.53 [0.46, 0.60] | 9.81 [8.50, 11.13] | 228.2 |
| group:R | 989,371 | 74.4768 | 0.07 [0.07, 0.08] | 32.85 [30.83, 34.90] | 1555.6 |
| group:K | 6,498 | 0.5352 | 0.32 [0.19, 0.46] | 1.03 [0.60, 1.51] | 324.9 |
| any | 995,869 | 75.012 | 0.07 [0.07, 0.08] | 33.88 [31.92, 36.02] | 1518.1 |
| all | 1,249,143 | 98.1093 | 0.07 [0.07, 0.07] | 42.25 [40.17, 44.48] | 1527.1 |

Classes of the missed golds: SEED_B0 51.65 [49.43, 53.82] (1,000); SEED_FORM 10.49 [9.09, 11.88] (203); GOLD_LINKED n/a (0); LINKED_ELSE 37.86 [35.80, 39.98] (733); ISOLATED 0.00 [0.00, 0.00] (0)

