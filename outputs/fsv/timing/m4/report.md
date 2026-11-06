# FSV timing, Family 2 (M4)

Family 2 is the cut zg6r (without topo_STRUCT, depth_STRUCT, WALK and WALKF; level L2) against the full zrckpn; Family 1, the full zg6r against the full zrckpn, is timed again beside it in the same job. All three versions run at the forms of amendments A2, A4, A6 and A7 (`docs/FSV_FAST_VERSIONS.md` sections 10 and 12 to 14) and, for the cut GNN, section 15 and its corrections 2 and 3. Times are ms per question at batch 1 on the host's CPU, cold unless labelled warm; each cell is p50 / p95 over the kept questions. Stage (a), the shared pool, is charged in full to every version: model-only = (b) + (c), end to end = (a) + (b) + (c). The headline is end to end; model-only is beside it. The three versions run on each question in an order that rotates, so each leads in turn.

Ratios are the first version / the second of the p50s, paired by question (by batch and round for the warm batches of 16), with the 2.5th and 97.5th percentiles over 2,000 resamples with replacement (`numpy.random.default_rng(0)`). A ratio under 1 means the first is faster. Nothing here is a call: the bars of section 5 are applied by a later file.

## BLAS 2 (the record)

### Ratios (first / second, p50s, paired)

| dataset | pair | end to end | model-only | (b) | (c) | warm b1 | warm b16 (A7's step) | warm b16 (module's step) |
|---|---|---|---|---|---|---|---|---|
| musique | cut GNN / MLP (Family 2) | 0.99 [0.98, 1.01] | 0.96 [0.94, 1.01] | 0.49 [0.47, 0.51] | 3.87 [3.69, 4.03] | 3.72 [3.56, 3.91] | 4.19 [3.99, 4.37] | 15.06 [14.36, 15.89] |
| musique | full GNN / MLP (Family 1) | 1.15 [1.11, 1.18] | 1.38 [1.33, 1.43] | 0.95 [0.91, 1.01] | 4.05 [3.85, 4.22] | 3.96 [3.82, 4.10] | 4.53 [4.31, 4.82] | 15.30 [14.51, 16.09] |
| musique | full GNN / cut GNN (the cut's gain) | 1.16 [1.12, 1.19] | 1.43 [1.36, 1.47] | 1.96 [1.86, 2.08] | 1.05 [1.01, 1.08] | 1.06 [1.02, 1.09] | 1.08 [1.05, 1.13] | 1.02 [0.98, 1.04] |
| hotpotqa | cut GNN / MLP (Family 2) | 0.98 [0.97, 1.01] | 0.90 [0.82, 0.97] | 0.50 [0.46, 0.59] | 2.56 [2.38, 2.74] | 2.49 [2.39, 2.57] | 2.76 [2.58, 2.91] | 5.23 [4.95, 5.53] |
| hotpotqa | full GNN / MLP (Family 1) | 1.00 [0.99, 1.02] | 1.04 [0.96, 1.12] | 0.68 [0.62, 0.76] | 2.85 [2.71, 3.04] | 2.82 [2.69, 2.92] | 3.05 [2.86, 3.22] | 5.65 [5.32, 5.96] |
| hotpotqa | full GNN / cut GNN (the cut's gain) | 1.02 [1.00, 1.04] | 1.16 [1.08, 1.26] | 1.35 [1.15, 1.50] | 1.11 [1.05, 1.20] | 1.13 [1.09, 1.17] | 1.11 [1.05, 1.17] | 1.08 [1.04, 1.11] |
| 2wiki | cut GNN / MLP (Family 2) | 1.00 [0.99, 1.01] | 0.97 [0.89, 1.02] | 0.62 [0.56, 0.66] | 2.62 [2.53, 2.75] | 2.55 [2.45, 2.61] | 2.70 [2.51, 2.96] | 5.03 [4.69, 5.47] |
| 2wiki | full GNN / MLP (Family 1) | 1.02 [1.00, 1.03] | 1.15 [1.06, 1.22] | 0.78 [0.72, 0.85] | 2.99 [2.85, 3.10] | 2.74 [2.68, 2.82] | 2.97 [2.73, 3.20] | 5.16 [4.81, 5.56] |
| 2wiki | full GNN / cut GNN (the cut's gain) | 1.02 [1.00, 1.03] | 1.19 [1.10, 1.30] | 1.26 [1.17, 1.40] | 1.14 [1.09, 1.18] | 1.08 [1.06, 1.12] | 1.10 [1.01, 1.17] | 1.03 [0.97, 1.09] |

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | version | (b) | (c) | model-only | end to end |
|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 125.8 / 182.0 | MLP (zrckpn) | 62.8 / 136.2 | 10.9 / 16.9 | 75.6 / 154.0 | 203.3 / 302.9 |
|  | | | | GNN, full (zg6r) | 59.9 / 123.1 | 44.3 / 71.1 | 104.4 / 190.5 | 234.1 / 349.0 |
|  | | | | GNN, cut (zg6r L2) | 30.6 / 59.0 | 42.3 / 68.5 | 72.8 / 127.0 | 201.5 / 291.7 |
| hotpotqa | 200 / 200 | 697 | 153.5 / 205.6 | MLP (zrckpn) | 9.8 / 18.3 | 2.0 / 5.1 | 12.1 / 22.2 | 167.2 / 216.5 |
|  | | | | GNN, full (zg6r) | 6.6 / 15.1 | 5.7 / 10.0 | 12.6 / 23.2 | 167.7 / 223.0 |
|  | | | | GNN, cut (zg6r L2) | 4.9 / 11.8 | 5.2 / 9.3 | 10.8 / 19.6 | 164.4 / 215.4 |
| 2wiki | 200 / 200 | 1074 | 168.6 / 207.1 | MLP (zrckpn) | 14.3 / 22.7 | 2.7 / 5.6 | 17.2 / 26.7 | 184.5 / 229.7 |
|  | | | | GNN, full (zg6r) | 11.2 / 18.5 | 8.2 / 11.8 | 19.9 / 29.3 | 188.1 / 232.1 |
|  | | | | GNN, cut (zg6r L2) | 8.9 / 15.9 | 7.2 / 11.3 | 16.7 / 25.0 | 184.2 / 228.9 |

### Parts of (b) (p50 ms) and the cut's gain per part (full GNN / cut GNN, paired)

| dataset | version | compile | rows | links | NEXT | inputs | gain: compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | MLP (zrckpn) | 35.13 | 18.52 | 2.21 | 3.61 | 2.40 |  |  |  |  |
|  | GNN, full (zg6r) | 34.20 | 18.22 | 2.13 | 0.00 | 4.71 |  |  |  |  |
|  | GNN, cut (zg6r L2) | 11.71 | 11.94 | 2.29 | 0.00 | 4.41 | 2.92 [2.70, 3.13] | 1.53 [1.44, 1.61] | 0.93 [0.81, 1.08] | 1.07 [0.97, 1.18] |
| hotpotqa | MLP (zrckpn) | 4.34 | 1.47 | 0.07 | 2.35 | 0.48 |  |  |  |  |
|  | GNN, full (zg6r) | 4.04 | 1.47 | 0.07 | 0.00 | 0.64 |  |  |  |  |
|  | GNN, cut (zg6r L2) | 2.92 | 1.08 | 0.07 | 0.00 | 0.54 | 1.39 [1.19, 1.59] | 1.36 [1.30, 1.49] | 1.00 [0.93, 1.03] | 1.19 [1.13, 1.24] |
| 2wiki | MLP (zrckpn) | 7.52 | 2.18 | 0.10 | 2.86 | 0.78 |  |  |  |  |
|  | GNN, full (zg6r) | 7.52 | 2.16 | 0.10 | 0.00 | 0.91 |  |  |  |  |
|  | GNN, cut (zg6r L2) | 5.31 | 1.59 | 0.10 | 0.00 | 0.85 | 1.42 [1.18, 1.63] | 1.36 [1.30, 1.42] | 0.98 [0.92, 1.02] | 1.06 [1.01, 1.12] |

### Stage (a), its share of each end-to-end time

| dataset | (a) p50 ms | share of MLP | share of full GNN | share of cut GNN |
|---|---|---|---|---|
| musique | 125.8 | 62% | 54% | 62% |
| hotpotqa | 153.5 | 92% | 92% | 93% |
| 2wiki | 168.6 | 91% | 90% | 92% |

### Warm (the forward alone, per question; labelled warm)

| dataset | version | b1 | b16, module's step | b16, A7's step |
|---|---|---|---|---|
| musique | MLP (zrckpn) | 11.9 / 18.0 | 12.0 / 14.8 | n/a |
|  | GNN, full (zg6r) | 47.1 / 80.7 | 183.9 / 276.5 | 54.5 / 74.4 |
|  | GNN, cut (zg6r L2) | 44.3 / 74.9 | 181.1 / 267.6 | 50.4 / 70.4 |
| hotpotqa | MLP (zrckpn) | 1.8 / 2.7 | 1.7 / 2.3 | n/a |
|  | GNN, full (zg6r) | 5.0 / 8.1 | 9.5 / 11.1 | 5.1 / 6.5 |
|  | GNN, cut (zg6r L2) | 4.4 / 7.5 | 8.8 / 10.7 | 4.7 / 5.7 |
| 2wiki | MLP (zrckpn) | 2.5 / 3.4 | 3.2 / 4.0 | n/a |
|  | GNN, full (zg6r) | 6.8 / 9.3 | 16.7 / 19.2 | 9.6 / 11.6 |
|  | GNN, cut (zg6r L2) | 6.3 / 8.4 | 16.2 / 19.7 | 8.7 / 11.0 |

### Checks, the forms that ran, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN / cut | max score diff MLP / GNN / cut | NEXT values differing | A7's step = module's (batches) | cut forms ran (runs) | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0 of 200 | 0 | 0 / 0 / 0 | 1.8e-05 / 9.5e-06 / 1e-05 | 6 | gnn 12 / 12; cut 12 / 12 | {'compile_cut': 234, 'walk_zero': 234, 'walkf_zero': 234} (234) | True | 3% / 2% / 53% | 11% / 6% / 62% |
| hotpotqa | 0 of 200 | 0 | 0 / 0 / 0 | 3.3e-06 / 5.4e-06 / 1.2e-05 | 1 | gnn 12 / 12; cut 12 / 12 | {'compile_cut': 234, 'walk_zero': 234, 'walkf_zero': 234} (234) | True | 18% / 7% / 68% | 8% / 3% / 46% |
| 2wiki | 0 of 200 | 0 | 0 / 0 / 0 | 3.8e-06 / 7.6e-06 / 1e-05 | 4 | gnn 12 / 12; cut 12 / 12 | {'compile_cut': 234, 'walk_zero': 234, 'walkf_zero': 234} (234) | True | 12% / 7% / 54% | 3% / 2% / 35% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | version | model-only: first / second half | led first / second / third | p95 / p50 of model-only |
|---|---|---|---|---|---|
| musique | 129 / 122 | MLP (zrckpn) | 77 / 73 | 78 / 74 / 70 | 2.04 |
|  | | GNN, full (zg6r) | 109 / 99 | 99 / 114 / 100 | 1.82 |
|  | | GNN, cut (zg6r L2) | 77 / 71 | 73 / 68 / 77 | 1.74 |
| hotpotqa | 153 / 154 | MLP (zrckpn) | 11 / 13 | 15 / 10 / 11 | 1.84 |
|  | | GNN, full (zg6r) | 12 / 13 | 16 / 12 / 11 | 1.84 |
|  | | GNN, cut (zg6r L2) | 10 / 11 | 14 / 10 / 10 | 1.82 |
| 2wiki | 170 / 168 | MLP (zrckpn) | 18 / 17 | 21 / 16 / 16 | 1.55 |
|  | | GNN, full (zg6r) | 21 / 19 | 24 / 19 / 17 | 1.47 |
|  | | GNN, cut (zg6r L2) | 17 / 16 | 19 / 14 / 15 | 1.50 |

## BLAS 1 (beside the record)

### Ratios (first / second, p50s, paired)

| dataset | pair | end to end | model-only | (b) | (c) | warm b1 | warm b16 (A7's step) | warm b16 (module's step) |
|---|---|---|---|---|---|---|---|---|
| musique | cut GNN / MLP (Family 2) | 0.99 [0.96, 1.02] | 0.99 [0.97, 1.03] | 0.50 [0.48, 0.52] | 3.74 [3.59, 3.96] | 3.68 [3.57, 3.88] | 4.47 [4.09, 4.61] | 16.00 [14.67, 16.96] |
| musique | full GNN / MLP (Family 1) | 1.15 [1.11, 1.18] | 1.39 [1.34, 1.47] | 0.96 [0.91, 1.00] | 4.00 [3.79, 4.23] | 3.79 [3.66, 4.00] | 4.72 [4.32, 4.89] | 16.12 [14.79, 16.65] |
| musique | full GNN / cut GNN (the cut's gain) | 1.17 [1.13, 1.19] | 1.40 [1.35, 1.48] | 1.93 [1.83, 2.02] | 1.07 [1.02, 1.10] | 1.03 [0.99, 1.06] | 1.06 [1.02, 1.10] | 1.01 [0.97, 1.03] |
| hotpotqa | cut GNN / MLP (Family 2) | 0.99 [0.99, 1.01] | 0.89 [0.85, 0.93] | 0.46 [0.43, 0.49] | 2.84 [2.67, 3.01] | 2.44 [2.34, 2.56] | 2.68 [2.49, 2.84] | 5.02 [4.61, 5.40] |
| hotpotqa | full GNN / MLP (Family 1) | 1.00 [0.99, 1.02] | 1.09 [1.04, 1.14] | 0.70 [0.66, 0.74] | 3.15 [2.99, 3.31] | 2.75 [2.64, 2.93] | 2.90 [2.64, 3.08] | 5.41 [4.98, 5.83] |
| hotpotqa | full GNN / cut GNN (the cut's gain) | 1.01 [1.00, 1.02] | 1.22 [1.17, 1.27] | 1.52 [1.44, 1.62] | 1.11 [1.06, 1.16] | 1.13 [1.09, 1.17] | 1.08 [1.01, 1.15] | 1.08 [1.01, 1.16] |
| 2wiki | cut GNN / MLP (Family 2) | 0.99 [0.98, 1.01] | 0.93 [0.89, 0.97] | 0.57 [0.54, 0.60] | 2.73 [2.60, 2.89] | 2.62 [2.49, 2.74] | 2.78 [2.64, 3.05] | 5.00 [4.72, 5.49] |
| 2wiki | full GNN / MLP (Family 1) | 1.01 [0.99, 1.03] | 1.14 [1.10, 1.19] | 0.77 [0.73, 0.80] | 3.06 [2.89, 3.24] | 2.86 [2.72, 2.96] | 3.07 [2.86, 3.36] | 5.26 [5.02, 5.80] |
| 2wiki | full GNN / cut GNN (the cut's gain) | 1.02 [1.00, 1.03] | 1.23 [1.18, 1.29] | 1.36 [1.29, 1.43] | 1.12 [1.06, 1.18] | 1.09 [1.04, 1.13] | 1.10 [1.03, 1.17] | 1.05 [1.01, 1.10] |

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | version | (b) | (c) | model-only | end to end |
|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 121.3 / 179.7 | MLP (zrckpn) | 61.0 / 136.1 | 11.4 / 16.7 | 74.0 / 149.6 | 199.5 / 306.8 |
|  | | | | GNN, full (zg6r) | 58.8 / 111.0 | 45.6 / 71.5 | 102.6 / 177.8 | 229.4 / 351.8 |
|  | | | | GNN, cut (zg6r L2) | 30.5 / 52.9 | 42.6 / 64.5 | 73.2 / 117.3 | 196.8 / 287.5 |
| hotpotqa | 200 / 200 | 697 | 154.2 / 205.6 | MLP (zrckpn) | 9.0 / 14.9 | 1.9 / 4.9 | 11.2 / 18.2 | 164.6 / 220.8 |
|  | | | | GNN, full (zg6r) | 6.3 / 10.3 | 6.0 / 9.7 | 12.2 / 19.7 | 164.9 / 220.8 |
|  | | | | GNN, cut (zg6r L2) | 4.1 / 8.4 | 5.4 / 9.0 | 10.0 / 17.4 | 163.3 / 217.6 |
| 2wiki | 200 / 200 | 1074 | 168.3 / 217.1 | MLP (zrckpn) | 13.2 / 19.8 | 2.8 / 5.5 | 16.4 / 24.2 | 186.5 / 236.4 |
|  | | | | GNN, full (zg6r) | 10.2 / 16.7 | 8.5 / 12.1 | 18.7 / 27.9 | 188.0 / 239.9 |
|  | | | | GNN, cut (zg6r L2) | 7.5 / 11.8 | 7.5 / 11.1 | 15.2 / 22.8 | 184.9 / 235.9 |

### Parts of (b) (p50 ms) and the cut's gain per part (full GNN / cut GNN, paired)

| dataset | version | compile | rows | links | NEXT | inputs | gain: compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | MLP (zrckpn) | 34.47 | 18.41 | 2.11 | 3.58 | 2.18 |  |  |  |  |
|  | GNN, full (zg6r) | 34.99 | 17.69 | 2.28 | 0.00 | 4.06 |  |  |  |  |
|  | GNN, cut (zg6r L2) | 11.80 | 11.75 | 2.19 | 0.00 | 4.08 | 2.97 [2.79, 3.08] | 1.51 [1.44, 1.60] | 1.04 [0.90, 1.14] | 0.99 [0.91, 1.06] |
| hotpotqa | MLP (zrckpn) | 3.58 | 1.46 | 0.07 | 2.46 | 0.49 |  |  |  |  |
|  | GNN, full (zg6r) | 3.77 | 1.43 | 0.07 | 0.00 | 0.62 |  |  |  |  |
|  | GNN, cut (zg6r L2) | 2.27 | 1.00 | 0.07 | 0.00 | 0.55 | 1.66 [1.57, 1.80] | 1.43 [1.34, 1.50] | 0.99 [0.95, 1.05] | 1.13 [1.05, 1.22] |
| 2wiki | MLP (zrckpn) | 6.78 | 2.20 | 0.10 | 2.93 | 0.75 |  |  |  |  |
|  | GNN, full (zg6r) | 6.75 | 2.17 | 0.09 | 0.00 | 0.88 |  |  |  |  |
|  | GNN, cut (zg6r L2) | 4.48 | 1.54 | 0.10 | 0.00 | 0.80 | 1.51 [1.38, 1.60] | 1.41 [1.33, 1.48] | 0.96 [0.93, 1.00] | 1.10 [1.07, 1.13] |

### Stage (a), its share of each end-to-end time

| dataset | (a) p50 ms | share of MLP | share of full GNN | share of cut GNN |
|---|---|---|---|---|
| musique | 121.3 | 61% | 53% | 62% |
| hotpotqa | 154.2 | 94% | 94% | 94% |
| 2wiki | 168.3 | 90% | 90% | 91% |

### Warm (the forward alone, per question; labelled warm)

| dataset | version | b1 | b16, module's step | b16, A7's step |
|---|---|---|---|---|
| musique | MLP (zrckpn) | 11.3 / 17.0 | 11.6 / 14.3 | n/a |
|  | GNN, full (zg6r) | 42.8 / 79.0 | 187.7 / 277.3 | 55.0 / 78.0 |
|  | GNN, cut (zg6r L2) | 41.6 / 67.2 | 186.4 / 268.2 | 52.0 / 67.8 |
| hotpotqa | MLP (zrckpn) | 1.8 / 3.8 | 1.8 / 2.5 | n/a |
|  | GNN, full (zg6r) | 5.1 / 8.0 | 9.7 / 12.7 | 5.2 / 6.7 |
|  | GNN, cut (zg6r L2) | 4.5 / 7.0 | 9.0 / 11.6 | 4.8 / 6.1 |
| 2wiki | MLP (zrckpn) | 2.5 / 5.1 | 3.1 / 3.7 | n/a |
|  | GNN, full (zg6r) | 7.2 / 11.1 | 16.4 / 19.7 | 9.6 / 11.1 |
|  | GNN, cut (zg6r L2) | 6.6 / 10.1 | 15.6 / 18.6 | 8.7 / 10.3 |

### Checks, the forms that ran, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN / cut | max score diff MLP / GNN / cut | NEXT values differing | A7's step = module's (batches) | cut forms ran (runs) | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | 24 of 200 | 0 | 16 / 22 / 5 | 0.0021 / 0.0013 / 0.00069 | 6 | gnn 12 / 12; cut 12 / 12 | {'compile_cut': 435, 'walk_zero': 435, 'walkf_zero': 435} (435) | True | 1% / 2% / 54% | 1% / 2% / 46% |
| hotpotqa | 5 of 200 | 0 | 3 / 2 / 3 | 0.0013 / 0.0013 / 0.00095 | 1 | gnn 12 / 12; cut 12 / 12 | {'compile_cut': 435, 'walk_zero': 435, 'walkf_zero': 435} (435) | True | 0% / 2% / 55% | 2% / 2% / 42% |
| 2wiki | 3 of 200 | 0 | 0 / 3 / 0 | 4e-05 / 0.00043 / 1e-05 | 4 | gnn 12 / 12; cut 12 / 12 | {'compile_cut': 435, 'walk_zero': 435, 'walkf_zero': 435} (435) | True | 15% / 5% / 64% | 2% / 2% / 42% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | version | model-only: first / second half | led first / second / third | p95 / p50 of model-only |
|---|---|---|---|---|---|
| musique | 123 / 119 | MLP (zrckpn) | 75 / 73 | 75 / 75 / 69 | 2.02 |
|  | | GNN, full (zg6r) | 108 / 99 | 96 / 111 / 107 | 1.73 |
|  | | GNN, cut (zg6r L2) | 75 / 71 | 74 / 70 / 78 | 1.60 |
| hotpotqa | 154 / 153 | MLP (zrckpn) | 11 / 11 | 12 / 11 / 11 | 1.63 |
|  | | GNN, full (zg6r) | 12 / 12 | 13 / 12 / 11 | 1.62 |
|  | | GNN, cut (zg6r L2) | 10 / 10 | 10 / 10 / 9 | 1.74 |
| 2wiki | 165 / 174 | MLP (zrckpn) | 17 / 16 | 17 / 16 / 16 | 1.48 |
|  | | GNN, full (zg6r) | 19 / 19 | 19 / 20 / 19 | 1.49 |
|  | | GNN, cut (zg6r L2) | 15 / 15 | 15 / 15 / 16 | 1.50 |

## Caveats

- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process is pinned to two logical processors (its own affinity, EcoQoS opted out, above-normal priority), the three versions rotate in leading each question, and the load of every logical processor is sampled for 1 s before and after the pass; the other jobs are not pinned and may share those processors.
- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed; PKO's prepared lists are read before timing as index-time data.
- Stage (a) is PKO's rule's own calls on one question at batch 1 on one thread, with the walks on P4b's records and only the orders the pool and the KT columns read (amendment A4); the layer step, stage (b)'s links, the MLP's NEXT columns and the encoder's segment z-scores are A4's and A6's (eager torch, one thread); OpenBLAS runs on 2 threads in the record and on 1 in the beside pass.
- The cut GNN does not compute topo_STRUCT, depth_STRUCT, WALK or WALKF (section 15): the topology, D and depth-basis laps of the compile and the two walk calls of the rows are left out, its pick has five blocks and its encoder is the cut fit's own. The forms that ran are counted in the checks table.
- Warm times are the forward alone on inputs built beforehand (one question or one batch at a time, outside the clock); b16 is the time of a batch of the first 192 questions over 16, over the rounds, and its ratios are paired by batch and round. The warm batches of 16 are timed with the module's layer step and with A7's batched step (byte for byte the module's); the MLP has one step.
