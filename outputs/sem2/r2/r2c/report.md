# SEM2 R2c Part A

Declared in docs/SEM2_EDGE_OPERATORS.md section 10. e3v verdict by recall at 20: **PARTIAL**; by ALL at 20: **PARTIAL**; ADDS on ALL at 20: musique, hotpotqa, 2wiki; Part B: **NAMED**.

## recall at 20 (points; paired bootstrap, 2,000 resamples, seed 0)

| arm/dataset | base | trav0 | trav | oracle | T1 = trav - base | T2 = trav - trav0 | trav - control | C = oracle - trav | ADDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e3v/multihoprag | 91.54 | 93.48 | 93.73 | 93.73 | +2.19 [+1.53, +2.92] ABOVE | +0.25 [-0.16, +0.67] LEVEL | +0.75 [+0.09, +1.46] ABOVE | +0.00 [+0.00, +0.00] LEVEL | False |
| e3v/squad | 95.28 | 95.33 | 95.19 | 95.23 | -0.08 [-0.32, +0.16] LEVEL | -0.13 [-0.30, +0.03] LEVEL | +0.79 [+0.53, +1.06] ABOVE | +0.04 [-0.02, +0.10] LEVEL | False |
| e3v/musique | 60.42 | 60.70 | 61.44 | 61.88 | +1.02 [+0.45, +1.57] ABOVE | +0.74 [+0.36, +1.13] ABOVE | +0.68 [+0.12, +1.20] ABOVE | +0.43 [+0.17, +0.73] ABOVE | True |
| e3v/metaqa | 1.25 | 1.45 | 1.38 | 2.07 | +0.13 [+0.01, +0.25] ABOVE | -0.07 [-0.19, +0.05] LEVEL | -0.08 [-0.20, +0.05] LEVEL | +0.69 [+0.54, +0.84] ABOVE | False |
| e3v/webqsp | 8.34 | 9.42 | 9.68 | 9.60 | +1.33 [+0.70, +2.00] ABOVE | +0.26 [-0.16, +0.71] LEVEL | -0.10 [-0.67, +0.51] LEVEL | -0.08 [-0.40, +0.21] LEVEL | False |
| e3v/hotpotqa | 77.59 | 77.73 | 78.20 | 78.18 | +0.61 [+0.34, +0.90] ABOVE | +0.47 [+0.30, +0.63] ABOVE | +1.52 [+1.20, +1.84] ABOVE | -0.01 [-0.14, +0.11] LEVEL | True |
| e3v/2wiki | 69.98 | 70.52 | 70.83 | 71.10 | +0.85 [+0.69, +1.00] ABOVE | +0.31 [+0.23, +0.40] ABOVE | +1.28 [+1.12, +1.46] ABOVE | +0.27 [+0.19, +0.34] ABOVE | True |
| frozen/multihoprag | 92.05 | 94.73 | 94.69 | 94.69 | +2.64 [+1.98, +3.33] ABOVE | -0.04 [-0.52, +0.41] LEVEL | +0.14 [-0.39, +0.71] LEVEL | +0.00 [+0.00, +0.00] LEVEL | False |
| frozen/squad | 96.39 | 96.09 | 96.26 | 96.28 | -0.13 [-0.35, +0.07] LEVEL | +0.17 [+0.02, +0.34] ABOVE | +0.07 [-0.17, +0.29] LEVEL | +0.02 [-0.03, +0.07] LEVEL | False |
| frozen/musique | 61.89 | 61.56 | 62.88 | 63.22 | +0.99 [+0.46, +1.48] ABOVE | +1.31 [+0.87, +1.74] ABOVE | +0.81 [+0.27, +1.32] ABOVE | +0.34 [+0.14, +0.53] ABOVE | True |
| frozen/metaqa | 1.48 | 1.59 | 2.24 | 2.33 | +0.77 [+0.59, +0.93] ABOVE | +0.65 [+0.48, +0.80] ABOVE | +0.32 [+0.16, +0.48] ABOVE | +0.09 [+0.04, +0.15] ABOVE | True |

## ALL at 20 (points; paired bootstrap, 2,000 resamples, seed 0)

| arm/dataset | base | trav0 | trav | oracle | T1 = trav - base | T2 = trav - trav0 | trav - control | C = oracle - trav | ADDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e3v/multihoprag | 78.05 | 83.89 | 84.69 | 84.69 | +6.64 [+4.87, +8.50] ABOVE | +0.80 [-0.35, +1.95] LEVEL | +3.19 [+1.50, +4.96] ABOVE | +0.00 [+0.00, +0.00] LEVEL | False |
| e3v/squad | 95.28 | 95.33 | 95.19 | 95.23 | -0.08 [-0.32, +0.16] LEVEL | -0.13 [-0.30, +0.03] LEVEL | +0.79 [+0.53, +1.06] ABOVE | +0.04 [-0.02, +0.10] LEVEL | False |
| e3v/musique | 24.91 | 25.45 | 26.81 | 27.18 | +1.90 [+0.99, +2.81] ABOVE | +1.37 [+0.70, +2.03] ABOVE | +1.41 [+0.50, +2.28] ABOVE | +0.37 [-0.08, +0.87] LEVEL | True |
| e3v/metaqa | 0.91 | 1.00 | 0.98 | 1.46 | +0.07 [-0.04, +0.18] LEVEL | -0.02 [-0.13, +0.09] LEVEL | -0.09 [-0.20, +0.03] LEVEL | +0.48 [+0.34, +0.61] ABOVE | False |
| e3v/webqsp | 5.92 | 7.12 | 7.05 | 6.99 | +1.13 [+0.53, +1.80] ABOVE | -0.07 [-0.47, +0.33] LEVEL | -0.27 [-0.80, +0.33] LEVEL | -0.07 [-0.33, +0.20] LEVEL | False |
| e3v/hotpotqa | 57.61 | 57.95 | 58.74 | 58.69 | +1.13 [+0.62, +1.66] ABOVE | +0.80 [+0.49, +1.09] ABOVE | +2.50 [+1.96, +3.09] ABOVE | -0.05 [-0.30, +0.19] LEVEL | True |
| e3v/2wiki | 38.92 | 39.84 | 40.32 | 40.79 | +1.40 [+1.11, +1.69] ABOVE | +0.49 [+0.32, +0.64] ABOVE | +1.86 [+1.55, +2.17] ABOVE | +0.47 [+0.31, +0.62] ABOVE | True |
| frozen/multihoprag | 79.65 | 86.37 | 86.46 | 86.46 | +6.81 [+5.13, +8.67] ABOVE | +0.09 [-1.15, +1.33] LEVEL | +0.88 [-0.53, +2.39] LEVEL | +0.00 [+0.00, +0.00] LEVEL | False |
| frozen/squad | 96.39 | 96.09 | 96.26 | 96.28 | -0.13 [-0.35, +0.07] LEVEL | +0.17 [+0.02, +0.34] ABOVE | +0.07 [-0.17, +0.29] LEVEL | +0.02 [-0.03, +0.07] LEVEL | False |
| frozen/musique | 27.72 | 27.76 | 29.54 | 30.04 | +1.82 [+0.83, +2.73] ABOVE | +1.78 [+1.03, +2.52] ABOVE | +0.41 [-0.46, +1.24] LEVEL | +0.50 [+0.12, +0.87] ABOVE | True |
| frozen/metaqa | 0.86 | 0.89 | 1.31 | 1.36 | +0.45 [+0.29, +0.60] ABOVE | +0.42 [+0.27, +0.57] ABOVE | +0.24 [+0.08, +0.39] ABOVE | +0.05 [+0.00, +0.11] LEVEL | True |
