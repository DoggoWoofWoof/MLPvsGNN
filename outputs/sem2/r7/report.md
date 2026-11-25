# SEM2 R7 report (links withheld; docs/SEM2_EDGE_OPERATORS.md section 15); 6 datasets read, 6 expected

Main call at f = 0.5: C_chain-point minus C_cosine, R-precision (points), paired bootstrap over units.

| dataset | units | reach of T2 by W | W alone | C_cosine | C_chain-point | pure chain-point | C_chain-point - C_cosine | control (unseen members) | R5 control (chain-point - chain-cos, unseen) | call |
|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | not read | | | | | | | | | |
| squad | 710 | 35.17 | 35.17 | 42.14 | 46.45 | 24.92 | +4.31 [+3.48, +5.11] ABOVE | +0.58 [+0.26, +0.91] ABOVE | +0.66 [+0.30, +1.08] ABOVE | ABOVE |
| musique | 900 | 28.54 | 28.54 | 32.24 | 33.59 | 9.25 | +1.35 [+0.77, +1.95] ABOVE | +0.39 [+0.05, +0.75] ABOVE | +0.61 [+0.20, +1.01] ABOVE | ABOVE |
| metaqa | 1200 | 26.9 | 26.9 | 28.1 | 28.72 | 3.29 | +0.62 [+0.18, +1.06] ABOVE | -0.05 [-0.23, +0.12] LEVEL | +0.15 [-0.05, +0.34] LEVEL | ABOVE |
| webqsp | 1200 | 24.99 | 24.99 | 26.07 | 26.69 | 2.52 | +0.62 [+0.05, +1.21] ABOVE | +0.16 [-0.09, +0.42] LEVEL | +0.30 [-0.00, +0.63] LEVEL | ABOVE |
| hotpotqa | 1200 | 27.59 | 27.59 | 29.55 | 30.2 | 4.22 | +0.65 [+0.36, +0.93] ABOVE | +0.19 [+0.02, +0.37] ABOVE | +0.35 [+0.15, +0.54] ABOVE | ABOVE |
| 2wiki | 1200 | 26.41 | 26.41 | 27.72 | 28.46 | 3.58 | +0.74 [+0.49, +1.01] ABOVE | +0.28 [+0.14, +0.41] ABOVE | +0.54 [+0.38, +0.70] ABOVE | ABOVE |

Main call ABOVE on 6 of 6 read, BELOW on 0 (rule 15.5: at least 4 above and none below, at least 4 read). Control (unseen members) ABOVE on 4, BELOW on 0: CONTROL_HOLDS = True. R5 under the control ABOVE on 4, BELOW on 0: R5_CONTROL_HOLDS = True. **FILL_CARRIES**.

## By f (R-precision, points): the walk, the fills, and the gain of chain-point over the cosine fill

| dataset | f | reach of T2 | W alone | C_cosine | C_chain-point | pure chain-point | C_chain-point - C_cosine | C_chain-point - W | pure chain-point - W |
|---|---|---|---|---|---|---|---|---|---|
| squad | 0.25 | 65.04 | 65.04 | 68.23 | 69.75 | 24.92 | +1.52 [+1.08, +1.96] ABOVE | +4.70 [+3.82, +5.72] ABOVE | -40.12 [-42.91, -37.14] BELOW |
| squad | 0.5 | 35.17 | 35.17 | 42.14 | 46.45 | 24.92 | +4.31 [+3.48, +5.11] ABOVE | +11.27 [+9.89, +12.78] ABOVE | -10.25 [-12.78, -7.54] BELOW |
| squad | 0.75 | 11.11 | 11.11 | 23.49 | 30.81 | 24.92 | +7.32 [+6.18, +8.39] ABOVE | +19.70 [+17.72, +21.77] ABOVE | +13.81 [+11.49, +16.11] ABOVE |
| squad | 0.9 | 1.99 | 1.99 | 16.97 | 25.82 | 24.92 | +8.85 [+7.50, +10.11] ABOVE | +23.83 [+21.47, +26.23] ABOVE | +22.93 [+20.55, +25.35] ABOVE |
| musique | 0.25 | 59.52 | 59.52 | 61.03 | 61.5 | 9.25 | +0.48 [+0.09, +0.88] ABOVE | +1.98 [+1.59, +2.44] ABOVE | -50.27 [-52.31, -48.25] BELOW |
| musique | 0.5 | 28.54 | 28.54 | 32.24 | 33.59 | 9.25 | +1.35 [+0.77, +1.95] ABOVE | +5.05 [+4.36, +5.77] ABOVE | -19.29 [-21.14, -17.57] BELOW |
| musique | 0.75 | 7.9 | 7.9 | 13.45 | 15.61 | 9.25 | +2.16 [+1.39, +2.96] ABOVE | +7.71 [+6.80, +8.62] ABOVE | +1.35 [+0.09, +2.58] ABOVE |
| musique | 0.9 | 1.53 | 1.53 | 7.79 | 10.49 | 9.25 | +2.70 [+1.84, +3.55] ABOVE | +8.96 [+7.96, +9.98] ABOVE | +7.72 [+6.62, +8.81] ABOVE |
| metaqa | 0.25 | 57.69 | 57.69 | 58.41 | 58.61 | 3.29 | +0.20 [-0.13, +0.55] LEVEL | +0.93 [+0.64, +1.26] ABOVE | -54.40 [-56.18, -52.43] BELOW |
| metaqa | 0.5 | 26.9 | 26.9 | 28.1 | 28.72 | 3.29 | +0.62 [+0.18, +1.06] ABOVE | +1.81 [+1.43, +2.25] ABOVE | -23.61 [-25.19, -22.03] BELOW |
| metaqa | 0.75 | 6.93 | 6.93 | 8.66 | 9.74 | 3.29 | +1.08 [+0.52, +1.65] ABOVE | +2.81 [+2.31, +3.33] ABOVE | -3.64 [-4.61, -2.68] BELOW |
| metaqa | 0.9 | 1.0 | 1.0 | 2.99 | 4.23 | 3.29 | +1.24 [+0.67, +1.84] ABOVE | +3.23 [+2.66, +3.81] ABOVE | +2.29 [+1.67, +2.91] ABOVE |
| webqsp | 0.25 | 56.31 | 56.31 | 57.09 | 57.21 | 2.52 | +0.12 [-0.32, +0.54] LEVEL | +0.90 [+0.58, +1.27] ABOVE | -53.79 [-55.67, -51.91] BELOW |
| webqsp | 0.5 | 24.99 | 24.99 | 26.07 | 26.69 | 2.52 | +0.62 [+0.05, +1.21] ABOVE | +1.70 [+1.21, +2.24] ABOVE | -22.47 [-24.02, -20.97] BELOW |
| webqsp | 0.75 | 6.12 | 6.12 | 7.49 | 8.37 | 2.52 | +0.88 [+0.21, +1.56] ABOVE | +2.25 [+1.69, +2.89] ABOVE | -3.60 [-4.55, -2.62] BELOW |
| webqsp | 0.9 | 1.08 | 1.08 | 2.57 | 3.53 | 2.52 | +0.95 [+0.26, +1.66] ABOVE | +2.44 [+1.84, +3.11] ABOVE | +1.44 [+0.77, +2.23] ABOVE |
| hotpotqa | 0.25 | 58.49 | 58.49 | 59.41 | 59.67 | 4.22 | +0.26 [+0.06, +0.47] ABOVE | +1.18 [+0.94, +1.44] ABOVE | -54.27 [-55.96, -52.73] BELOW |
| hotpotqa | 0.5 | 27.59 | 27.59 | 29.55 | 30.2 | 4.22 | +0.65 [+0.36, +0.93] ABOVE | +2.61 [+2.21, +3.03] ABOVE | -23.37 [-24.87, -21.96] BELOW |
| hotpotqa | 0.75 | 7.63 | 7.63 | 10.26 | 11.23 | 4.22 | +0.96 [+0.58, +1.35] ABOVE | +3.60 [+3.14, +4.07] ABOVE | -3.41 [-4.29, -2.54] BELOW |
| hotpotqa | 0.9 | 1.36 | 1.36 | 4.22 | 5.45 | 4.22 | +1.23 [+0.85, +1.63] ABOVE | +4.09 [+3.59, +4.60] ABOVE | +2.85 [+2.29, +3.43] ABOVE |
| 2wiki | 0.25 | 59.34 | 59.34 | 59.96 | 60.22 | 3.58 | +0.26 [+0.08, +0.46] ABOVE | +0.88 [+0.69, +1.09] ABOVE | -55.76 [-57.29, -54.18] BELOW |
| 2wiki | 0.5 | 26.41 | 26.41 | 27.72 | 28.46 | 3.58 | +0.74 [+0.49, +1.01] ABOVE | +2.05 [+1.76, +2.35] ABOVE | -22.83 [-24.16, -21.55] BELOW |
| 2wiki | 0.75 | 6.74 | 6.74 | 8.56 | 9.91 | 3.58 | +1.35 [+1.01, +1.69] ABOVE | +3.18 [+2.79, +3.56] ABOVE | -3.15 [-3.98, -2.40] BELOW |
| 2wiki | 0.9 | 0.99 | 0.99 | 3.02 | 4.53 | 3.58 | +1.50 [+1.14, +1.85] ABOVE | +3.53 [+3.13, +3.93] ABOVE | +2.59 [+2.12, +3.04] ABOVE |

## The crossover (the smallest grid f at which chain-point alone is not BELOW the stored walk alone)

| dataset | f | crossover |
|---|---|---|
| squad | 0.25, 0.5, 0.75, 0.9 | 0.75 |
| musique | 0.25, 0.5, 0.75, 0.9 | 0.75 |
| metaqa | 0.25, 0.5, 0.75, 0.9 | 0.9 |
| webqsp | 0.25, 0.5, 0.75, 0.9 | 0.9 |
| hotpotqa | 0.25, 0.5, 0.75, 0.9 | 0.9 |
| 2wiki | 0.25, 0.5, 0.75, 0.9 | 0.9 |

## recall at 4m: C_chain-point - C_cosine by f

| dataset | f = 0.25 | f = 0.5 | f = 0.75 | f = 0.9 |
|---|---|---|---|---|
| squad | +3.17 [+2.61, +3.79] ABOVE | +7.08 [+6.04, +8.22] ABOVE | +10.45 [+9.12, +11.85] ABOVE | +11.88 [+10.44, +13.40] ABOVE |
| musique | +1.79 [+1.22, +2.40] ABOVE | +3.83 [+2.95, +4.67] ABOVE | +5.43 [+4.30, +6.49] ABOVE | +6.11 [+4.91, +7.24] ABOVE |
| metaqa | +2.05 [+1.50, +2.69] ABOVE | +4.09 [+3.30, +4.94] ABOVE | +5.40 [+4.48, +6.45] ABOVE | +5.72 [+4.75, +6.79] ABOVE |
| webqsp | +1.57 [+0.94, +2.20] ABOVE | +3.17 [+2.28, +4.08] ABOVE | +4.08 [+3.01, +5.09] ABOVE | +4.37 [+3.26, +5.43] ABOVE |
| hotpotqa | +1.05 [+0.69, +1.40] ABOVE | +2.08 [+1.56, +2.59] ABOVE | +2.90 [+2.30, +3.54] ABOVE | +3.19 [+2.55, +3.84] ABOVE |
| 2wiki | +1.28 [+1.01, +1.58] ABOVE | +2.47 [+2.07, +2.90] ABOVE | +3.37 [+2.86, +3.86] ABOVE | +3.68 [+3.16, +4.18] ABOVE |

## R-precision, unseen members: C_chain-point - C_cosine by f

| dataset | f = 0.25 | f = 0.5 | f = 0.75 | f = 0.9 |
|---|---|---|---|---|
| squad | +0.31 [+0.10, +0.55] ABOVE | +0.58 [+0.26, +0.91] ABOVE | +0.82 [+0.46, +1.19] ABOVE | +0.96 [+0.58, +1.34] ABOVE |
| musique | +0.25 [+0.06, +0.51] ABOVE | +0.39 [+0.05, +0.75] ABOVE | +0.43 [+0.02, +0.86] ABOVE | +0.47 [+0.05, +0.91] ABOVE |
| metaqa | -0.11 [-0.27, +0.04] LEVEL | -0.05 [-0.23, +0.12] LEVEL | -0.06 [-0.27, +0.14] LEVEL | -0.11 [-0.34, +0.11] LEVEL |
| webqsp | -0.00 [-0.21, +0.20] LEVEL | +0.16 [-0.09, +0.42] LEVEL | +0.19 [-0.12, +0.50] LEVEL | +0.20 [-0.12, +0.53] LEVEL |
| hotpotqa | +0.17 [+0.05, +0.29] ABOVE | +0.19 [+0.02, +0.37] ABOVE | +0.24 [+0.03, +0.47] ABOVE | +0.32 [+0.10, +0.57] ABOVE |
| 2wiki | +0.10 [-0.00, +0.20] LEVEL | +0.28 [+0.14, +0.41] ABOVE | +0.48 [+0.30, +0.68] ABOVE | +0.51 [+0.30, +0.72] ABOVE |

## recall at 4m, unseen members: C_chain-point - C_cosine by f

| dataset | f = 0.25 | f = 0.5 | f = 0.75 | f = 0.9 |
|---|---|---|---|---|
| squad | +0.76 [+0.50, +1.08] ABOVE | +1.48 [+1.07, +1.91] ABOVE | +1.95 [+1.50, +2.43] ABOVE | +2.12 [+1.66, +2.60] ABOVE |
| musique | +0.48 [+0.16, +0.83] ABOVE | +1.02 [+0.59, +1.49] ABOVE | +1.33 [+0.76, +1.88] ABOVE | +1.41 [+0.82, +1.99] ABOVE |
| metaqa | +0.50 [+0.22, +0.81] ABOVE | +0.86 [+0.51, +1.27] ABOVE | +1.02 [+0.62, +1.47] ABOVE | +1.05 [+0.64, +1.50] ABOVE |
| webqsp | +0.64 [+0.26, +1.02] ABOVE | +1.15 [+0.63, +1.68] ABOVE | +1.38 [+0.80, +1.96] ABOVE | +1.43 [+0.85, +2.01] ABOVE |
| hotpotqa | +0.40 [+0.21, +0.62] ABOVE | +0.60 [+0.34, +0.87] ABOVE | +0.83 [+0.53, +1.15] ABOVE | +0.90 [+0.58, +1.24] ABOVE |
| 2wiki | +0.48 [+0.32, +0.65] ABOVE | +0.85 [+0.63, +1.06] ABOVE | +1.05 [+0.78, +1.33] ABOVE | +1.12 [+0.83, +1.42] ABOVE |

## Other fills (R-precision), C_chain-point minus C_chain-cos and minus C_compose, at f = 0.5

| dataset | minus C_chain-cos | minus C_compose |
|---|---|---|
| squad | +3.58 [+2.79, +4.35] ABOVE | +2.40 [+1.75, +3.04] ABOVE |
| musique | +1.53 [+0.96, +2.09] ABOVE | +0.83 [+0.31, +1.35] ABOVE |
| metaqa | +0.98 [+0.60, +1.39] ABOVE | +0.34 [-0.12, +0.80] LEVEL |
| webqsp | +0.75 [+0.21, +1.32] ABOVE | -0.41 [-0.87, +0.01] LEVEL |
| hotpotqa | +0.66 [+0.39, +0.90] ABOVE | +0.44 [+0.19, +0.71] ABOVE |
| 2wiki | +0.86 [+0.65, +1.09] ABOVE | +0.30 [+0.06, +0.54] ABOVE |

## C_chain-point - C_cosine by set size at f = 0.5 (R-precision, points)

| dataset | m 2-3 | m 4-10 | m 11-100 |
|---|---|---|---|
| squad | 46: +0.72 [-1.45, +2.90] LEVEL | 81: +2.78 [+0.40, +5.27] ABOVE | 583: +4.81 [+3.88, +5.73] ABOVE |
| musique | 81: -0.41 [-2.27, +1.44] LEVEL | 180: -0.52 [-1.89, +0.87] LEVEL | 639: +2.11 [+1.38, +2.85] ABOVE |
| metaqa | 415: +1.16 [+0.16, +2.17] ABOVE | 390: +0.30 [-0.33, +0.94] LEVEL | 395: +0.36 [-0.02, +0.73] LEVEL |
| webqsp | 317: +0.47 [-0.68, +1.79] LEVEL | 492: -0.13 [-0.79, +0.44] LEVEL | 391: +1.69 [+0.56, +2.99] ABOVE |
| hotpotqa | 166: -0.10 [-1.11, +1.00] LEVEL | 258: +0.40 [-0.27, +1.09] LEVEL | 776: +0.89 [+0.54, +1.20] ABOVE |
| 2wiki | 139: -0.48 [-1.68, +0.72] LEVEL | 218: +0.35 [-0.37, +1.12] LEVEL | 843: +1.05 [+0.82, +1.31] ABOVE |

## Unseen share of T2 (members with no pair of two TRAIN nodes), %

| dataset | unseen share |
|---|---|
| squad | 32.29 |
| musique | 32.16 |
| metaqa | 33.86 |
| webqsp | 33.72 |
| hotpotqa | 32.7 |
| 2wiki | 35.3 |
