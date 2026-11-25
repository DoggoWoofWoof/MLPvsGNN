# SEM2 R4 report (sets, not points; docs/SEM2_EDGE_OPERATORS.md section 12); 7 of 7 datasets read

| dataset | units | mean m | lam* | R-prec cosine | point | shift | region | region - point | region - shift | point - cosine | call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 46 | 62.72 | 1.0 | 34.85 | 41.92 | 41.75 | 41.75 | -0.17 [-0.36, +0.01] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +7.07 [+4.55, +9.28] ABOVE | LEVEL |
| squad | 448 | 35.81 | 1.0 | 36.92 | 48.61 | 48.3 | 48.3 | -0.31 [-0.76, +0.12] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +11.69 [+9.93, +13.40] ABOVE | LEVEL |
| musique | 600 | 14.92 | 0.5 | 11.06 | 22.19 | 19.82 | 18.52 | -3.66 [-4.79, -2.59] BELOW | -1.30 [-2.19, -0.46] BELOW | +11.12 [+9.21, +12.95] ABOVE | BELOW |
| metaqa | 1433 | 4.57 | 1.0 | 5.74 | 16.68 | 15.65 | 15.65 | -1.02 [-1.71, -0.30] BELOW | +0.00 [+0.00, +0.00] LEVEL | +10.94 [+9.82, +12.08] ABOVE | BELOW |
| webqsp | 2639 | 5.93 | 1.0 | 7.84 | 38.51 | 37.9 | 37.9 | -0.61 [-1.18, -0.01] BELOW | +0.00 [+0.00, +0.00] LEVEL | +30.67 [+29.29, +32.10] ABOVE | BELOW |
| hotpotqa | 800 | 8.02 | 1.0 | 12.59 | 17.82 | 16.02 | 16.02 | -1.80 [-2.41, -1.16] BELOW | +0.00 [+0.00, +0.00] LEVEL | +5.23 [+4.21, +6.25] ABOVE | BELOW |
| 2wiki | 800 | 11.03 | 1.0 | 8.95 | 14.98 | 12.71 | 12.71 | -2.27 [-2.90, -1.66] BELOW | +0.00 [+0.00, +0.00] LEVEL | +6.03 [+5.13, +6.92] ABOVE | BELOW |

Region above point (R-precision, 95% interval above 0) on 0 of 7; below on 5. Rule (12.6): at least 4 of 7 ABOVE and none BELOW. **NOT_CARRIED**.

## By set size (region - point, R-precision, points)

| dataset | m 2-3 | m 4-10 | m 11-100 |
|---|---|---|---|
| multihoprag | - | - | 46: -0.17 [-0.36, +0.01] LEVEL |
| squad | 28: -4.17 [-8.93, +0.00] LEVEL | 45: -1.12 [-3.66, +1.18] LEVEL | 375: +0.07 [-0.20, +0.34] LEVEL |
| musique | 168: -3.17 [-5.65, -0.89] BELOW | 188: -3.46 [-6.04, -1.11] BELOW | 244: -4.16 [-5.46, -2.90] BELOW |
| metaqa | 956: +0.24 [-0.75, +1.26] LEVEL | 339: -2.23 [-3.26, -1.29] BELOW | 138: -6.86 [-8.29, -5.61] BELOW |
| webqsp | 1489: -0.41 [-1.37, +0.49] LEVEL | 827: -0.33 [-1.30, +0.61] LEVEL | 323: -2.20 [-3.13, -1.35] BELOW |
| hotpotqa | 276: -2.29 [-3.68, -0.79] BELOW | 335: -2.36 [-3.25, -1.52] BELOW | 189: -0.06 [-0.55, +0.43] LEVEL |
| 2wiki | 220: -2.73 [-4.32, -1.29] BELOW | 300: -2.96 [-4.06, -1.93] BELOW | 280: -1.16 [-1.73, -0.58] BELOW |

## Recall at 4m beside (points)

| dataset | region - point | region - cosine |
|---|---|---|
| multihoprag | -0.15 [-0.55, +0.25] LEVEL | +4.72 [+2.11, +7.34] ABOVE |
| squad | -0.51 [-0.92, -0.11] BELOW | +10.94 [+9.06, +12.76] ABOVE |
| musique | -3.91 [-5.25, -2.65] BELOW | +14.26 [+12.10, +16.38] ABOVE |
| metaqa | +0.61 [-0.07, +1.34] LEVEL | +16.88 [+15.22, +18.68] ABOVE |
| webqsp | +0.61 [-0.08, +1.31] LEVEL | +34.63 [+33.05, +36.12] ABOVE |
| hotpotqa | -3.72 [-4.67, -2.86] BELOW | +7.78 [+6.66, +8.97] ABOVE |
| 2wiki | -4.25 [-5.12, -3.42] BELOW | +6.61 [+5.59, +7.56] ABOVE |
