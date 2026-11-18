# CCS: chain-completing selection over the GNN's top five (report)

Written by `outputs/mp_unified/zccs.py report` under docs/CCS_CHAIN_COMPLETING_SELECTION.md (section 6's rule). Base = each re-read's own top five (p@swa); R@5 in points, three-seed cells, paired 95% bootstrap.

## Verdicts

- **BF: NOT_ADOPTED**
- **BJ: NOT_ADOPTED**
- Carried: none

## G1 (identity against the record's reads)

- exact: 36

## Section 4 first: reach and bounds (families S + NER; all three in report.json)

| cell | partly-found q | adj1 gold / non-gold (lift) | adj2 gold / non-gold (lift) | O2 | O2b | O1s | unlinked T rows gold | linked T rows gold |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| zg6r musique in-domain | 602 | 63.1 / 3.7 (17.0) | 36.7 / 0.9 (41.1) | 20.32 | 11.82 | 10.32 | 15.1 | 25.9 |
| zg6r hotpotqa in-domain | 81 | 71.5 / 17.3 (4.1) | 47.8 / 8.5 (5.7) | 3.02 | 2.02 | 1.92 | 13.2 | 27.8 |
| zg6r 2wiki in-domain | 50 | 91.9 / 16.1 (5.8) | 45.6 / 6.6 (7.0) | 1.82 | 1.06 | 1.06 | 17.5 | 40.1 |
| zg6r musique held-out | 705 | 65.2 / 7.8 (8.4) | 39.7 / 1.5 (26.3) | 26.41 | 16.08 | 13.86 | 14.9 | 20.0 |
| zg6r hotpotqa held-out | 106 | 70.8 / 16.7 (4.2) | 48.1 / 8.6 (5.6) | 3.98 | 2.70 | 2.57 | 12.1 | 28.2 |
| gnx musique in-domain | 598 | 63.7 / 3.5 (18.5) | 37.7 / 0.8 (45.8) | 20.34 | 11.96 | 10.36 | 14.4 | 26.4 |
| gnx hotpotqa in-domain | 85 | 71.4 / 17.3 (4.1) | 50.4 / 8.8 (5.7) | 3.17 | 2.23 | 2.13 | 12.9 | 27.5 |
| gnx 2wiki in-domain | 48 | 95.1 / 17.8 (5.4) | 46.1 / 7.6 (6.2) | 1.80 | 1.03 | 1.03 | 17.0 | 40.1 |
| gnx musique held-out | 679 | 65.0 / 6.8 (9.6) | 39.6 / 1.4 (28.3) | 24.74 | 14.99 | 12.92 | 14.5 | 21.5 |
| gnx hotpotqa held-out | 99 | 73.4 / 16.6 (4.4) | 49.3 / 8.2 (6.0) | 3.87 | 2.60 | 2.48 | 12.8 | 28.1 |

## The calls (R@5, form minus base)

| cell | base R@5 | BF | BJ |
| --- | ---: | --- | --- |
| zg6r musique in-domain | 69.0 | -1.03 [-1.54, -0.49] BELOW | -0.38 [-0.64, -0.12] BELOW |
| zg6r hotpotqa in-domain | 94.3 | -1.77 [-2.32, -1.23] BELOW | -0.17 [-0.40, +0.08] LEVEL |
| zg6r 2wiki in-domain | 94.9 | -1.45 [-1.87, -1.08] BELOW | -0.22 [-0.42, -0.04] BELOW |
| zg6r musique held-out | 60.8 | -1.01 [-1.46, -0.54] BELOW | -0.23 [-0.48, +0.03] LEVEL |
| zg6r hotpotqa held-out | 92.9 | -1.78 [-2.38, -1.18] BELOW | -0.18 [-0.43, +0.08] LEVEL |
| gnx musique in-domain | 69.1 | -1.07 [-1.54, -0.57] BELOW | -0.43 [-0.73, -0.15] BELOW |
| gnx hotpotqa in-domain | 94.1 | -1.95 [-2.53, -1.37] BELOW | -0.12 [-0.33, +0.08] LEVEL |
| gnx 2wiki in-domain | 95.0 | -1.43 [-1.86, -1.03] BELOW | -0.17 [-0.33, -0.05] BELOW |
| gnx musique held-out | 63.1 | -1.33 [-1.82, -0.83] BELOW | -0.09 [-0.33, +0.14] LEVEL |
| gnx hotpotqa held-out | 93.2 | -2.12 [-2.77, -1.47] BELOW | -0.10 [-0.33, +0.15] LEVEL |

## Beside: FC@5, firing and outcomes

| cell | form | FC@5 diff | fired (%) | swaps per fit: gold in / non-gold out | non-gold in / gold out | both gold | neither |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| zg6r musique in-domain | BF | -0.23 [-1.10, +0.60] LEVEL | 49.2 | 26.0 | 64.0 | 3.3 | 398.3 |
| zg6r musique in-domain | BJ | -0.30 [-0.83, +0.17] LEVEL | 18.3 | 8.0 | 20.7 | 0.7 | 153.3 |
| zg6r hotpotqa in-domain | BF | -3.43 [-4.53, -2.37] BELOW | 28.2 | 6.3 | 41.7 | 0.0 | 234.3 |
| zg6r hotpotqa in-domain | BJ | -0.33 [-0.80, +0.17] LEVEL | 5.9 | 3.0 | 6.3 | 0.0 | 49.7 |
| zg6r 2wiki in-domain | BF | -2.40 [-3.22, -1.67] BELOW | 30.1 | 3.7 | 37.3 | 0.2 | 260.3 |
| zg6r 2wiki in-domain | BJ | -0.37 [-0.77, +0.00] LEVEL | 6.3 | 2.2 | 6.8 | 0.0 | 54.5 |
| zg6r musique held-out | BF | -0.30 [-0.97, +0.37] LEVEL | 35.1 | 18.7 | 54.3 | 3.3 | 275.0 |
| zg6r musique held-out | BJ | +0.13 [-0.23, +0.53] LEVEL | 12.3 | 7.7 | 17.7 | 1.7 | 96.0 |
| zg6r hotpotqa held-out | BF | -3.27 [-4.43, -2.10] BELOW | 28.2 | 9.3 | 45.0 | 0.0 | 227.7 |
| zg6r hotpotqa held-out | BJ | -0.37 [-0.87, +0.17] LEVEL | 6.0 | 3.3 | 7.0 | 0.0 | 49.7 |
| gnx musique in-domain | BF | -0.50 [-1.37, +0.30] LEVEL | 49.4 | 22.3 | 60.0 | 4.0 | 408.0 |
| gnx musique in-domain | BJ | -0.27 [-0.80, +0.23] LEVEL | 18.8 | 8.3 | 23.3 | 2.7 | 153.3 |
| gnx hotpotqa in-domain | BF | -3.80 [-4.97, -2.63] BELOW | 28.9 | 6.3 | 45.3 | 0.0 | 237.7 |
| gnx hotpotqa in-domain | BJ | -0.23 [-0.67, +0.17] LEVEL | 5.8 | 2.3 | 4.7 | 0.0 | 50.7 |
| gnx 2wiki in-domain | BF | -2.47 [-3.30, -1.72] BELOW | 28.3 | 3.0 | 34.8 | 0.7 | 244.2 |
| gnx 2wiki in-domain | BJ | -0.32 [-0.63, -0.05] BELOW | 5.4 | 0.8 | 4.2 | 0.0 | 49.2 |
| gnx musique held-out | BF | -0.47 [-1.17, +0.20] LEVEL | 37.7 | 16.0 | 60.7 | 2.3 | 297.7 |
| gnx musique held-out | BJ | +0.33 [-0.10, +0.73] LEVEL | 12.3 | 8.3 | 14.3 | 4.0 | 96.7 |
| gnx hotpotqa held-out | BF | -4.10 [-5.37, -2.83] BELOW | 25.5 | 7.7 | 50.0 | 0.0 | 197.7 |
| gnx hotpotqa held-out | BJ | -0.23 [-0.67, +0.23] LEVEL | 4.9 | 3.0 | 5.0 | 0.0 | 40.7 |

## Beside: PH1's groups (R@5 and FC@5, form minus base)

| cell | group | questions | base R@5 | BF R@5 | BJ R@5 | BF FC@5 | BJ FC@5 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| zg6r musique in-domain | 2 hops | 518 | 79.0 | +0.52 [-0.19, +1.19] LEVEL | +0.10 [-0.10, +0.32] LEVEL | +1.09 [-0.06, +2.19] LEVEL | +0.13 [-0.26, +0.58] LEVEL |
| zg6r musique in-domain | 3 hops | 316 | 64.2 | -2.71 [-3.80, -1.76] BELOW | -1.05 [-1.76, -0.39] BELOW | -2.64 [-4.64, -0.74] BELOW | -1.16 [-2.64, +0.21] LEVEL |
| zg6r musique in-domain | 4 hops | 166 | 46.6 | -2.66 [-3.77, -1.61] BELOW | -0.55 [-1.21, +0.10] LEVEL | +0.20 [+0.00, +0.60] LEVEL | +0.00 [+0.00, +0.00] LEVEL |
| zg6r hotpotqa in-domain | bridge | 811 | 93.0 | -0.10 [-0.51, +0.31] LEVEL | +0.10 [-0.08, +0.33] LEVEL | -0.08 [-0.86, +0.74] LEVEL | +0.21 [-0.16, +0.66] LEVEL |
| zg6r hotpotqa in-domain | comparison | 189 | 99.9 | -8.91 [-11.11, -6.79] BELOW | -1.32 [-2.38, -0.53] BELOW | -17.81 [-22.22, -13.58] BELOW | -2.65 [-4.76, -1.06] BELOW |
| zg6r 2wiki in-domain | compositional | 413 | 94.7 | +0.26 [+0.04, +0.50] ABOVE | +0.18 [+0.00, +0.42] LEVEL | +0.53 [+0.08, +1.01] ABOVE | +0.36 [+0.00, +0.85] LEVEL |
| zg6r 2wiki in-domain | inference | 108 | 93.8 | +0.23 [+0.00, +0.69] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.46 [+0.00, +1.39] LEVEL | +0.00 [+0.00, +0.00] LEVEL |
| zg6r 2wiki in-domain | comparison | 244 | 100.0 | -5.57 [-7.04, -4.27] BELOW | -1.16 [-1.91, -0.55] BELOW | -11.13 [-14.07, -8.54] BELOW | -2.32 [-3.83, -1.09] BELOW |
| zg6r 2wiki in-domain | bridge_comparison | 235 | 90.5 | -0.97 [-1.45, -0.53] BELOW | -0.05 [-0.25, +0.11] LEVEL | +0.21 [+0.00, +0.57] LEVEL | +0.21 [+0.00, +0.57] LEVEL |
| zg6r musique held-out | 2 hops | 518 | 72.0 | +0.29 [-0.26, +0.84] LEVEL | +0.26 [-0.06, +0.61] LEVEL | +0.52 [-0.52, +1.48] LEVEL | +0.52 [-0.06, +1.16] LEVEL |
| zg6r musique held-out | 3 hops | 316 | 54.0 | -2.43 [-3.45, -1.55] BELOW | -0.60 [-1.09, -0.14] BELOW | -1.90 [-3.48, -0.74] BELOW | -0.53 [-1.27, +0.00] LEVEL |
| zg6r musique held-out | 4 hops | 166 | 38.7 | -2.36 [-3.51, -1.35] BELOW | -1.05 [-1.91, -0.30] BELOW | +0.20 [+0.00, +0.60] LEVEL | +0.20 [+0.00, +0.60] LEVEL |
| zg6r hotpotqa held-out | bridge | 811 | 91.2 | +0.00 [-0.45, +0.41] LEVEL | +0.16 [-0.02, +0.39] LEVEL | +0.37 [-0.45, +1.15] LEVEL | +0.33 [-0.04, +0.78] LEVEL |
| zg6r hotpotqa held-out | comparison | 189 | 100.0 | -9.44 [-11.99, -7.32] BELOW | -1.68 [-2.82, -0.70] BELOW | -18.87 [-23.99, -14.64] BELOW | -3.35 [-5.64, -1.41] BELOW |
| gnx musique in-domain | 2 hops | 518 | 79.2 | +0.32 [-0.32, +0.90] LEVEL | +0.13 [-0.19, +0.42] LEVEL | +0.58 [-0.52, +1.67] LEVEL | +0.19 [-0.39, +0.77] LEVEL |
| gnx musique in-domain | 3 hops | 316 | 64.3 | -2.71 [-3.76, -1.69] BELOW | -1.12 [-1.76, -0.53] BELOW | -2.11 [-4.01, -0.42] BELOW | -0.84 [-2.11, +0.32] LEVEL |
| gnx musique in-domain | 4 hops | 166 | 46.6 | -2.31 [-3.41, -1.31] BELOW | -0.85 [-1.61, -0.10] BELOW | -0.80 [-2.21, +0.00] LEVEL | -0.60 [-1.81, +0.00] LEVEL |
| gnx hotpotqa in-domain | bridge | 811 | 92.7 | -0.14 [-0.53, +0.25] LEVEL | +0.12 [+0.00, +0.27] ABOVE | -0.16 [-0.90, +0.57] LEVEL | +0.25 [+0.00, +0.53] ABOVE |
| gnx hotpotqa in-domain | comparison | 189 | 100.0 | -9.70 [-12.08, -7.32] BELOW | -1.15 [-2.21, -0.35] BELOW | -19.40 [-24.16, -14.64] BELOW | -2.29 [-4.41, -0.70] BELOW |
| gnx 2wiki in-domain | compositional | 413 | 95.0 | +0.16 [-0.06, +0.38] LEVEL | +0.06 [+0.00, +0.16] LEVEL | +0.32 [-0.12, +0.77] LEVEL | +0.12 [+0.00, +0.32] LEVEL |
| gnx 2wiki in-domain | inference | 108 | 93.2 | +0.23 [+0.00, +0.62] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.46 [+0.00, +1.23] LEVEL | +0.00 [+0.00, +0.00] LEVEL |
| gnx 2wiki in-domain | comparison | 244 | 100.0 | -5.57 [-7.21, -4.17] BELOW | -0.82 [-1.40, -0.34] BELOW | -11.13 [-14.41, -8.33] BELOW | -1.64 [-2.80, -0.68] BELOW |
| gnx 2wiki in-domain | bridge_comparison | 235 | 90.6 | -0.69 [-1.14, -0.28] BELOW | +0.02 [-0.03, +0.11] LEVEL | +0.28 [+0.00, +0.71] LEVEL | +0.14 [+0.00, +0.43] LEVEL |
| gnx musique held-out | 2 hops | 518 | 74.5 | +0.06 [-0.52, +0.64] LEVEL | +0.42 [+0.10, +0.80] ABOVE | +0.06 [-0.97, +1.09] LEVEL | +0.77 [+0.13, +1.54] ABOVE |
| gnx musique held-out | 3 hops | 316 | 56.3 | -2.88 [-3.87, -1.93] BELOW | -0.63 [-1.09, -0.25] BELOW | -1.58 [-2.96, -0.32] BELOW | -0.21 [-1.06, +0.42] LEVEL |
| gnx musique held-out | 4 hops | 166 | 40.3 | -2.71 [-3.82, -1.71] BELOW | -0.65 [-1.31, -0.10] BELOW | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL |
| gnx hotpotqa held-out | bridge | 811 | 91.7 | -0.25 [-0.76, +0.21] LEVEL | +0.04 [-0.18, +0.29] LEVEL | -0.33 [-1.27, +0.53] LEVEL | +0.04 [-0.41, +0.53] LEVEL |
| gnx hotpotqa held-out | comparison | 189 | 99.9 | -10.14 [-12.70, -7.76] BELOW | -0.70 [-1.59, -0.09] BELOW | -20.28 [-25.40, -15.52] BELOW | -1.41 [-3.17, -0.18] BELOW |
