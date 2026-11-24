# UL2P report: the label precision ledger on the e3v layout

Declared in docs/UL2P_LABEL_PRECISION_E3V.md. Written 2026-11-24T20:00:06Z. Decides nothing.

## Expectations

| id | call | reads | detail |
|---|---|---|---|
| E1 | HELD | 1 of 1 |  |
| E2 | NOT_READ | 0 of 0 |  |
| E3 | NOT_READ | 0 of 0 |  |
| E4 | NOT_HELD | 6 of 7 | multihoprag/RK/GP: HM vs any LEVEL |
| E5 | NOT_HELD | 0 of 1 | multihoprag/RK/GP: NTc first vs last quintile LEVEL |
| E6 | HELD | 1 of 1 | multihoprag/RK: multi minus single ABOVE: held |
| E7 | NOT_READ | 0 of 0 |  |

## What follows (section 6: names only; no rule changes)

- Candidate label weights (a label that POINTS and is ABOVE any link): multihoprag/RK/GP: label:K.knn
- Candidate tighter cuts (a form BELOW any link whose first quintile is ABOVE it): multihoprag/RK/GP: H5
- Flags (a label BELOW any link holding at least 5% of the any-link candidates): none

## multihoprag


### multihoprag / RK / s1eval / GP: 1,130 questions, 2,747 slots

| unit | pairs | price per anchor | precision % | reach % | graph price | lift call | vs any (points) |
|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0124 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | BELOW_BASE | -1.77 [-1.84, -1.70] BELOW |
| label:R.name_comention | 25,005 | 88.1273 | 1.77 [1.70, 1.84] | 84.89 [83.13, 86.73] | 10.7 | POINTS | 0.01 [-0.00, 0.01] LEVEL |
| label:R.title_comention | 75 | 0.2573 | 2.08 [0.80, 3.70] | 0.29 [0.11, 0.52] | 9.4 | POINTS | 0.32 [-0.97, 1.94] LEVEL |
| label:K.knn | 1,366 | 5.2027 | 9.96 [9.26, 10.72] | 28.14 [26.25, 30.28] | 1.8 | POINTS | 8.19 [7.51, 8.93] ABOVE |
| group:R | 25,080 | 88.3846 | 1.77 [1.71, 1.85] | 85.18 [83.46, 86.99] | 10.7 | POINTS | 0.01 [0.00, 0.01] ABOVE |
| group:K | 263 | 0.6861 | 0.98 [0.47, 1.62] | 0.36 [0.18, 0.61] | 26.3 | POINTS | -0.79 [-1.32, -0.14] BELOW |
| any | 25,343 | 89.0707 | 1.77 [1.70, 1.84] | 85.55 [83.86, 87.32] | 10.8 | POINTS | - |
| form:H5 | 2,387 | 10.9755 | 1.05 [0.87, 1.24] | 6.26 [5.20, 7.39] | 13.9 | POINTS | -0.72 [-0.92, -0.50] BELOW |
| form:HM | 1,332 | 5.4951 | 1.85 [1.53, 2.19] | 5.53 [4.50, 6.58] | 8.8 | POINTS | 0.09 [-0.27, 0.44] LEVEL |
| form:HK1 | 2,712 | 12.131 | 0.75 [0.61, 0.90] | 4.95 [4.00, 5.89] | 19.9 | POINTS | -1.02 [-1.20, -0.84] BELOW |
| form:NTc | 13,668 | 43.4429 | 0.27 [0.22, 0.31] | 6.26 [5.24, 7.31] | 79.5 | NO_LIFT | -1.50 [-1.59, -1.41] BELOW |
| form:NT1 | 4,604 | 15.0613 | 0.29 [0.22, 0.36] | 2.33 [1.77, 2.93] | 71.9 | NO_LIFT | -1.48 [-1.59, -1.38] BELOW |
| form:HK2 | 4,886 | 21.4536 | 0.60 [0.50, 0.71] | 7.03 [5.87, 8.19] | 25.3 | POINTS | -1.17 [-1.30, -1.03] BELOW |
| form:H10 | 4,588 | 20.1427 | 0.71 [0.61, 0.83] | 7.83 [6.61, 9.08] | 21.3 | POINTS | -1.05 [-1.20, -0.90] BELOW |

Multi minus single: 10.03 [9.23, 10.90] ABOVE.

Forms, first and last rank quintile (precision %):

- H5: first 3.01 [2.45, 3.61] (266 targets), last 0.06 [0.00, 0.18] (2 targets), ABOVE
- HM: first 4.58 [3.64, 5.67] (216 targets), last 0.39 [0.00, 0.85] (6 targets), ABOVE
- HK1: first 1.93 [1.52, 2.36] (190 targets), last 0.04 [0.00, 0.12] (2 targets), ABOVE
- NTc: first 0.12 [0.06, 0.19] (28 targets), last 0.08 [0.04, 0.13] (24 targets), LEVEL
- NT1: first 0.90 [0.67, 1.15] (112 targets), last 0.03 [0.00, 0.08] (2 targets), ABOVE
- HK2: first 1.93 [1.61, 2.27] (354 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE
- H10: first 1.99 [1.63, 2.35] (326 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE

### multihoprag / RK / s1sel / GP: 224 questions, 504 slots

| unit | pairs | price per anchor | precision % | reach % | graph price | lift call | vs any (points) |
|---|---|---|---|---|---|---|---|
| label:R.mention | 2 | 0.0087 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | BELOW_BASE | -1.71 [-1.86, -1.56] BELOW |
| label:R.name_comention | 25,005 | 85.4904 | 1.72 [1.57, 1.88] | 83.93 [80.08, 87.48] | 59.1 | POINTS | 0.01 [-0.00, 0.02] LEVEL |
| label:R.title_comention | 75 | 0.2713 | 1.28 [0.00, 4.32] | 0.20 [0.00, 0.63] | 75.0 | NO_LIFT | -0.43 [-1.83, 2.64] LEVEL |
| label:K.knn | 1,366 | 5.1374 | 9.41 [7.87, 11.02] | 27.58 [23.31, 32.03] | 9.8 | POINTS | 7.70 [6.17, 9.29] ABOVE |
| group:R | 25,080 | 85.7617 | 1.72 [1.56, 1.87] | 84.13 [80.32, 87.63] | 59.2 | POINTS | 0.01 [-0.00, 0.01] LEVEL |
| group:K | 263 | 0.6226 | 0.56 [0.00, 1.86] | 0.20 [0.00, 0.66] | 263.0 | NO_LIFT | -1.15 [-1.83, 0.18] LEVEL |
| any | 25,343 | 86.3843 | 1.71 [1.56, 1.86] | 84.33 [80.56, 87.83] | 59.6 | POINTS | - |
| form:H5 | 2,387 | 10.3722 | 1.31 [0.88, 1.79] | 7.74 [5.29, 10.49] | 61.2 | POINTS | -0.40 [-0.90, 0.15] LEVEL |
| form:HM | 1,332 | 5.5304 | 2.45 [1.69, 3.27] | 7.74 [5.29, 10.50] | 34.2 | POINTS | 0.74 [-0.08, 1.61] LEVEL |
| form:HK1 | 2,712 | 11.1513 | 0.72 [0.45, 1.03] | 4.56 [2.84, 6.49] | 117.9 | POINTS | -0.99 [-1.34, -0.61] BELOW |
| form:NTc | 13,668 | 43.1322 | 0.30 [0.20, 0.40] | 7.34 [4.95, 9.89] | 369.4 | NO_LIFT | -1.41 [-1.61, -1.22] BELOW |
| form:NT1 | 4,604 | 14.9339 | 0.54 [0.31, 0.77] | 4.56 [2.64, 6.54] | 200.2 | POINTS | -1.18 [-1.48, -0.87] BELOW |
| form:HK2 | 4,886 | 19.9861 | 0.75 [0.52, 1.00] | 8.53 [5.96, 11.31] | 113.6 | POINTS | -0.96 [-1.28, -0.64] BELOW |
| form:H10 | 4,588 | 19.033 | 0.80 [0.56, 1.08] | 8.73 [6.08, 11.68] | 104.3 | POINTS | -0.91 [-1.25, -0.54] BELOW |

Multi minus single: 9.39 [7.56, 11.28] ABOVE.

Forms, first and last rank quintile (precision %):

- H5: first 4.17 [2.82, 5.64] (70 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE
- HM: first 6.97 [4.51, 9.80] (62 targets), last 0.63 [0.00, 2.12] (2 targets), ABOVE
- HK1: first 2.00 [1.14, 2.96] (36 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE
- NTc: first 0.05 [0.00, 0.14] (2 targets), last 0.00 [0.00, 0.00] (0 targets), LEVEL
- NT1: first 1.68 [0.90, 2.56] (38 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE
- HK2: first 2.17 [1.44, 2.99] (72 targets), last 0.24 [0.00, 0.61] (4 targets), ABOVE
- H10: first 2.43 [1.64, 3.27] (74 targets), last 0.00 [0.00, 0.00] (0 targets), ABOVE

## squad

n/a: one gold per question: no gold pair, so the gold-anchored ledger has no cell (section 9, D-P1)
