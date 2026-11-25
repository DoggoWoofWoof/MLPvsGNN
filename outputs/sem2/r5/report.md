# SEM2 R5 report (branching; docs/SEM2_EDGE_OPERATORS.md section 13); 6 of 7 datasets read, 1 not a dataset for this read

| dataset | units | mean m | mean T1 | R-prec cosine | chain-cos | chain-point | compose | chain-point - chain-cos | chain-point - compose | chain-point - cosine | chain-cos - cosine | call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0 | | | | | | | | | | | not a dataset for this read (0 test units in all, fewer than 20) |
| squad | 710 | 36.89 | 25.91 | 15.7 | 17.25 | 24.92 | 20.44 | +7.68 [+6.40, +9.02] ABOVE | +4.48 [+3.53, +5.51] ABOVE | +9.23 [+7.84, +10.57] ABOVE | +1.55 [+0.92, +2.15] ABOVE | ABOVE |
| musique | 900 | 32.02 | 8.71 | 6.43 | 6.11 | 9.25 | 7.66 | +3.14 [+2.32, +3.94] ABOVE | +1.59 [+0.87, +2.32] ABOVE | +2.82 [+1.93, +3.66] ABOVE | -0.32 [-0.72, +0.07] LEVEL | ABOVE |
| metaqa | 1200 | 12.98 | 8.36 | 2.03 | 1.34 | 3.29 | 2.56 | +1.94 [+1.43, +2.50] ABOVE | +0.72 [+0.10, +1.39] ABOVE | +1.26 [+0.68, +1.87] ABOVE | -0.69 [-0.96, -0.43] BELOW | ABOVE |
| webqsp | 1200 | 11.13 | 16.51 | 1.5 | 1.4 | 2.52 | 3.07 | +1.12 [+0.43, +1.85] ABOVE | -0.54 [-1.08, -0.03] BELOW | +1.02 [+0.32, +1.76] ABOVE | -0.10 [-0.37, +0.16] LEVEL | ABOVE |
| hotpotqa | 1200 | 25.72 | 7.18 | 2.93 | 2.88 | 4.22 | 3.65 | +1.34 [+0.98, +1.66] ABOVE | +0.57 [+0.22, +0.92] ABOVE | +1.29 [+0.90, +1.70] ABOVE | -0.04 [-0.36, +0.27] LEVEL | ABOVE |
| 2wiki | 1200 | 33.9 | 7.54 | 2.1 | 1.9 | 3.58 | 3.03 | +1.68 [+1.40, +1.98] ABOVE | +0.55 [+0.24, +0.86] ABOVE | +1.48 [+1.11, +1.85] ABOVE | -0.20 [-0.44, +0.03] LEVEL | ABOVE |

chain-point above chain-cos (R-precision, 95% interval above 0) on 6 of 6 read; below on 0. Rule (13.6): at least 4 above and none below, at least 4 read. **BRANCH_CARRIES**.

## Recall at 4m beside (chain-point - chain-cos, chain-point - cosine)

| dataset | chain-point - chain-cos | chain-point - cosine |
|---|---|---|
| squad | +10.19 [+8.76, +11.65] ABOVE | +12.14 [+10.66, +13.65] ABOVE |
| musique | +6.22 [+5.07, +7.29] ABOVE | +6.22 [+4.99, +7.38] ABOVE |
| metaqa | +6.38 [+5.39, +7.48] ABOVE | +5.82 [+4.86, +6.92] ABOVE |
| webqsp | +4.51 [+3.47, +5.59] ABOVE | +4.51 [+3.39, +5.59] ABOVE |
| hotpotqa | +3.22 [+2.66, +3.75] ABOVE | +3.30 [+2.66, +3.95] ABOVE |
| 2wiki | +3.60 [+3.08, +4.09] ABOVE | +3.72 [+3.20, +4.21] ABOVE |

## Branches (the first hop, in points)

| dataset | scorer | branch recall | branch precision | any true first hop |
|---|---|---|---|---|
| squad | chain-cos | 53.67 | 43.69 | 79.15 |
| squad | chain-point | 63.38 | 50.69 | 86.62 |
| musique | chain-cos | 23.72 | 11.34 | 45.33 |
| musique | chain-point | 34.95 | 17.17 | 59.67 |
| metaqa | chain-cos | 0.99 | 0.63 | 3.58 |
| metaqa | chain-point | 2.54 | 1.77 | 11.33 |
| webqsp | chain-cos | 30.34 | 27.69 | 52.58 |
| webqsp | chain-point | 29.96 | 26.84 | 52.33 |
| hotpotqa | chain-cos | 20.03 | 9.01 | 47.0 |
| hotpotqa | chain-point | 28.62 | 12.83 | 66.17 |
| 2wiki | chain-cos | 16.5 | 8.33 | 42.08 |
| 2wiki | chain-point | 25.85 | 12.68 | 61.75 |

## chain-point - chain-cos, by path kind (R-precision, points)

| dataset | R then R | S then S | mixed |
|---|---|---|---|
| squad | 710: +7.68 [+6.40, +9.02] ABOVE | - | - |
| musique | 900: +3.14 [+2.32, +3.94] ABOVE | - | - |
| metaqa | 200: +1.09 [+0.60, +1.65] ABOVE | - | 1000: +2.12 [+1.52, +2.78] ABOVE |
| webqsp | 100: +0.06 [-0.33, +0.53] LEVEL | - | 1100: +1.22 [+0.43, +2.03] ABOVE |
| hotpotqa | 600: +0.88 [+0.51, +1.24] ABOVE | 100: +2.79 [+1.93, +3.72] ABOVE | 500: +1.59 [+0.98, +2.26] ABOVE |
| 2wiki | 600: +1.19 [+0.82, +1.61] ABOVE | 100: +4.11 [+3.08, +5.19] ABOVE | 500: +1.79 [+1.32, +2.26] ABOVE |

## chain-point - chain-cos, by set size (R-precision, points)

| dataset | m 11-100 | m 2-3 | m 4-10 |
|---|---|---|---|
| squad | 583: +7.25 [+5.95, +8.53] ABOVE | 46: +9.78 [+2.90, +17.39] ABOVE | 81: +9.56 [+4.79, +14.73] ABOVE |
| musique | 639: +3.70 [+2.63, +4.76] ABOVE | 81: +0.82 [-1.23, +3.09] LEVEL | 180: +2.20 [+0.78, +3.71] ABOVE |
| metaqa | 395: +1.21 [+0.64, +1.85] ABOVE | 415: +2.93 [+1.77, +4.18] ABOVE | 390: +1.63 [+0.88, +2.44] ABOVE |
| webqsp | 391: +2.75 [+1.20, +4.37] ABOVE | 317: +0.89 [-0.47, +2.37] LEVEL | 492: -0.03 [-0.85, +0.73] LEVEL |
| hotpotqa | 776: +1.76 [+1.40, +2.10] ABOVE | 166: -0.40 [-1.41, +0.60] LEVEL | 258: +1.18 [+0.25, +2.09] ABOVE |
| 2wiki | 843: +1.90 [+1.62, +2.19] ABOVE | 139: +0.48 [-0.48, +1.56] LEVEL | 218: +1.62 [+0.85, +2.53] ABOVE |

## chain-point - chain-cos, by T1 size (R-precision, points)

| dataset | |T1| over 10 | |T1| up to 10 |
|---|---|---|
| squad | 423: +7.52 [+5.92, +9.24] ABOVE | 287: +7.91 [+5.94, +10.01] ABOVE |
| musique | 201: +4.96 [+3.46, +6.62] ABOVE | 699: +2.62 [+1.59, +3.62] ABOVE |
| metaqa | 287: +4.39 [+3.05, +5.80] ABOVE | 913: +1.17 [+0.68, +1.72] ABOVE |
| webqsp | 542: +0.89 [-0.20, +1.93] LEVEL | 658: +1.31 [+0.35, +2.33] ABOVE |
| hotpotqa | 249: +2.00 [+1.46, +2.61] ABOVE | 951: +1.16 [+0.74, +1.56] ABOVE |
| 2wiki | 267: +1.06 [+0.66, +1.47] ABOVE | 933: +1.86 [+1.51, +2.21] ABOVE |
