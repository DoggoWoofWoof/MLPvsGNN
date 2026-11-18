# SEM1: a graph-smoothed document space as a first-stage channel (s1eval)

Declared in docs/SEM1_GRAPH_SPACE.md. Recall in %; differences in points with 95% intervals (paired, 2,000 resamples, seed 0).

Chosen settings (s1sel): frozen h2_a0.1; qwen3e06 h1_a0.25; mpnet h1_a0.1; mpnetqa h2_a0.1

## musique

| arm | base R@5 | add R@5 | ctrl R@5 | swap R@5 | u R@1000 | v R@1000 | C1 add - base | C2 add - ctrl | C3 add - frozen base |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 47.34 | 47.61 | 47.64 | 47.51 | 87.48 | 87.25 | +0.27 [-0.19, +0.72] LEVEL | -0.03 [-0.22, +0.17] LEVEL |  |
| qwen3e06 | 44.98 | 44.90 | 44.22 | 45.29 | 84.17 | 83.60 | -0.09 [-0.48, +0.31] LEVEL | +0.67 [+0.41, +0.92] ABOVE | -2.44 [-3.15, -1.73] BELOW |
| mpnet | 41.24 | 39.80 | 39.69 | 41.64 | 80.57 | 79.97 | -1.44 [-1.86, -1.00] BELOW | +0.11 [-0.06, +0.28] LEVEL | -7.54 [-8.40, -6.65] BELOW |
| mpnetqa | 40.92 | 39.41 | 39.42 | 40.94 | 80.61 | 79.80 | -1.51 [-1.94, -1.06] BELOW | -0.01 [-0.21, +0.19] LEVEL | -7.93 [-8.80, -7.03] BELOW |

- qwen3e06: 349 golds lost against the frozen base top 5; add wins back 35 (0.1003); add drops 91 of the arm's own base top-5 golds.
- mpnet: 572 golds lost against the frozen base top 5; add wins back 13 (0.0227); add drops 151 of the arm's own base top-5 golds.
- mpnetqa: 589 golds lost against the frozen base top 5; add wins back 20 (0.034); add drops 157 of the arm's own base top-5 golds.

## squad

| arm | base R@5 | add R@5 | ctrl R@5 | swap R@5 | u R@1000 | v R@1000 | C1 add - base | C2 add - ctrl | C3 add - frozen base |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 90.53 | 90.26 | 90.30 | 90.47 | 99.75 | 99.76 | -0.28 [-0.52, -0.04] BELOW | -0.04 [-0.14, +0.06] LEVEL |  |
| qwen3e06 | 87.17 | 85.66 | 85.88 | 86.71 | 99.06 | 99.15 | -1.51 [-1.80, -1.23] BELOW | -0.22 [-0.40, -0.05] BELOW | -4.87 [-5.33, -4.43] BELOW |
| mpnet | 83.98 | 81.62 | 81.66 | 83.88 | 98.71 | 98.73 | -2.36 [-2.70, -2.04] BELOW | -0.03 [-0.16, +0.08] LEVEL | -8.91 [-9.49, -8.36] BELOW |
| mpnetqa | 85.02 | 82.86 | 82.97 | 84.86 | 99.02 | 99.01 | -2.16 [-2.46, -1.84] BELOW | -0.11 [-0.24, +0.01] LEVEL | -7.67 [-8.19, -7.13] BELOW |

- qwen3e06: 518 golds lost against the frozen base top 5; add wins back 26 (0.0502); add drops 235 of the arm's own base top-5 golds.
- mpnet: 879 golds lost against the frozen base top 5; add wins back 13 (0.0148); add drops 340 of the arm's own base top-5 golds.
- mpnetqa: 765 golds lost against the frozen base top 5; add wins back 14 (0.0183); add drops 293 of the arm's own base top-5 golds.

## metaqa

| arm | base R@5 | add R@5 | ctrl R@5 | swap R@5 | u R@1000 | v R@1000 | C1 add - base | C2 add - ctrl | C3 add - frozen base |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 0.47 | 0.75 | 0.76 | 0.49 | 16.29 | 15.34 | +0.28 [+0.19, +0.38] ABOVE | -0.00 [-0.04, +0.02] LEVEL |  |
| qwen3e06 | 0.35 | 0.39 | 0.39 | 0.36 | 12.93 | 10.90 | +0.03 [-0.01, +0.09] LEVEL | -0.00 [-0.03, +0.02] LEVEL | -0.09 [-0.18, +0.01] LEVEL |
| mpnet | 0.54 | 0.63 | 0.66 | 0.47 | 18.52 | 18.27 | +0.09 [+0.02, +0.16] ABOVE | -0.02 [-0.06, +0.00] LEVEL | +0.16 [+0.04, +0.27] ABOVE |
| mpnetqa | 0.46 | 0.47 | 0.49 | 0.44 | 14.12 | 13.87 | +0.02 [-0.03, +0.06] LEVEL | -0.02 [-0.05, +0.01] LEVEL | -0.00 [-0.10, +0.09] LEVEL |

- qwen3e06: 66 golds lost against the frozen base top 5; add wins back 2 (0.0303); add drops 10 of the arm's own base top-5 golds.
- mpnet: 63 golds lost against the frozen base top 5; add wins back 4 (0.0635); add drops 7 of the arm's own base top-5 golds.
- mpnetqa: 59 golds lost against the frozen base top 5; add wins back 3 (0.0508); add drops 3 of the arm's own base top-5 golds.

## Expectations (section 8)

- **Z1_frozen**: NOT_HELD
- **Z1_qwen3e06**: NOT_HELD
- **Z1_mpnet**: NOT_HELD
- **Z1_mpnetqa**: NOT_HELD
- **Z2**: NOT_HELD
- **Z3**: NOT_HELD

D98 (open; the user's call). Nothing is adopted.
