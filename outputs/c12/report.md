Cold, batch 1, total per question, p50 ms; ratios with 2,000-resample bootstrap 95% CIs. Each model's fastest form is the faster of its C12 and C11 paths by cold p50.

universal family:

| dataset | umlp C12 | umlp C11 | gnn C12 | gnn C11 | **gnn / umlp (fastest forms)** | at least 2x | C12 on both | C11 on both |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2wiki | 4.11 | 4.25 | 6.00 | 6.26 | **1.46 [1.43, 1.50]** (gnn12 / umlp12) | no | 1.46 [1.43, 1.50] | 1.47 [1.42, 1.52] |
| hotpotqa | 3.91 | 4.13 | 5.88 | 6.04 | **1.50 [1.44, 1.56]** (gnn12 / umlp12) | no | 1.50 [1.44, 1.56] | 1.46 [1.42, 1.52] |
| metaqa | 49.47 | 51.55 | 77.45 | 79.26 | **1.57 [1.55, 1.57]** (gnn12 / umlp12) | no | 1.57 [1.55, 1.57] | 1.54 [1.52, 1.55] |
| musique | 75.66 | 80.86 | 143.26 | 148.47 | **1.89 [1.87, 1.91]** (gnn12 / umlp12) | no | 1.89 [1.87, 1.91] | 1.84 [1.82, 1.85] |
| squad | 2.69 | 2.67 | 4.24 | 4.48 | **1.59 [1.55, 1.69]** (gnn12 / umlp11) | no | 1.58 [1.53, 1.69] | 1.68 [1.58, 1.77] |
| webqsp | 69.87 | 72.75 | 107.26 | 109.15 | **1.54 [1.51, 1.55]** (gnn12 / umlp12) | no | 1.54 [1.52, 1.55] | 1.50 [1.48, 1.53] |

lean family:

| dataset | zrc C12 | zrc C11 | zsp C12 | zsp C11 | **zsp / zrc (fastest forms)** | at least 2x | C12 on both | C11 on both |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2wiki | 2.52 | 2.53 | 2.61 | 2.71 | **1.04 [1.01, 1.07]** (zsp12 / zrc12) | no | 1.04 [1.01, 1.07] | 1.07 [1.03, 1.11] |
| hotpotqa | 2.41 | 2.36 | 2.52 | 2.48 | **1.05 [1.01, 1.09]** (zsp10 / zrc11) | no | 1.04 [1.01, 1.07] | 1.05 [1.01, 1.09] |
| metaqa | 26.42 | 26.44 | 28.07 | 28.16 | **1.06 [1.06, 1.07]** (zsp12 / zrc12) | no | 1.06 [1.06, 1.07] | 1.06 [1.06, 1.07] |
| musique | 35.87 | 36.01 | 42.19 | 42.23 | **1.18 [1.16, 1.19]** (zsp12 / zrc12) | no | 1.18 [1.16, 1.19] | 1.17 [1.16, 1.18] |
| squad | 1.65 | 1.66 | 1.73 | 1.77 | **1.05 [1.03, 1.09]** (zsp12 / zrc12) | no | 1.05 [1.03, 1.10] | 1.06 [1.03, 1.10] |
| webqsp | 84.93 | 82.65 | 79.62 | 79.65 | **0.96 [0.93, 0.99]** (zsp12 / zrc11) | no | 0.94 [0.92, 0.97] | 0.96 [0.94, 0.99] |

Across the families (fastest forms):

| dataset | fastest GNN | fastest GNN / universal MLP | fastest GNN / zrc | universal GNN / zrc |
| --- | --- | --- | --- | --- |
| 2wiki | zsp12 | 0.63 [0.62, 0.65] | 1.04 [1.01, 1.07] | 2.38 [2.33, 2.45] |
| hotpotqa | zsp10 | 0.63 [0.61, 0.66] | 1.05 [1.01, 1.09] | 2.49 [2.38, 2.57] |
| metaqa | zsp12 | 0.57 [0.56, 0.57] | 1.06 [1.06, 1.07] | 2.93 [2.91, 2.95] |
| musique | zsp12 | 0.56 [0.55, 0.56] | 1.18 [1.16, 1.19] | 3.99 [3.94, 4.03] |
| squad | zsp12 | 0.65 [0.64, 0.68] | 1.05 [1.03, 1.09] | 2.57 [2.50, 2.72] |
| webqsp | zsp12 | 1.14 [1.09, 1.19] | 0.96 [0.93, 0.99] | 1.30 [1.24, 1.34] |

What C12's kernels give each model (C11 path / C12 path, cold total; and the compile alone):

| dataset | twin0 total | twin0 compile | gnn0 total | gnn0 compile | zrc total | zrc compile | zsp total | zsp compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 1.03 [1.00, 1.07] | 1.05 [1.03, 1.10] | 1.04 [1.01, 1.07] | 1.08 [1.05, 1.13] | 1.00 [0.98, 1.03] | 1.01 [0.97, 1.05] | 1.04 [0.99, 1.08] | 1.05 [1.00, 1.09] |
| hotpotqa | 1.06 [1.02, 1.10] | 1.08 [1.04, 1.12] | 1.03 [0.99, 1.08] | 1.07 [1.04, 1.12] | 0.98 [0.95, 1.02] | 0.98 [0.96, 1.01] | 0.98 [0.95, 1.03] | 0.97 [0.93, 1.02] |
| metaqa | 1.04 [1.03, 1.05] | 1.06 [1.04, 1.08] | 1.02 [1.01, 1.03] | 1.06 [1.04, 1.07] | 1.00 [1.00, 1.01] | 1.01 [1.00, 1.02] | 1.00 [0.99, 1.01] | 1.01 [1.00, 1.02] |
| musique | 1.07 [1.06, 1.08] | 1.10 [1.09, 1.11] | 1.04 [1.03, 1.04] | 1.10 [1.09, 1.11] | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.01] | 1.00 [0.99, 1.01] | 1.01 [1.00, 1.02] |
| squad | 0.99 [0.97, 1.02] | 1.02 [0.98, 1.04] | 1.06 [0.99, 1.11] | 1.04 [1.01, 1.09] | 1.01 [0.99, 1.03] | 1.00 [0.98, 1.02] | 1.02 [0.98, 1.06] | 1.01 [0.97, 1.03] |
| webqsp | 1.04 [1.02, 1.06] | 1.06 [1.04, 1.08] | 1.02 [1.00, 1.04] | 1.06 [1.04, 1.08] | 0.97 [0.95, 1.02] | 1.00 [0.98, 1.02] | 1.00 [0.97, 1.04] | 1.02 [1.00, 1.04] |

Cold p95 / p99 ms:

- 2wiki: umlp12 5.69 / 6.44; gnn12 8.03 / 8.90; zrc12 3.44 / 3.87; zsp12 3.94 / 4.88; umlp11 6.09 / 6.87; gnn11 8.00 / 9.64; zrc11 3.38 / 3.71; zsp10 3.77 / 4.22
- hotpotqa: umlp12 5.54 / 6.55; gnn12 7.89 / 8.62; zrc12 3.26 / 4.19; zsp12 3.54 / 4.18; umlp11 5.82 / 6.71; gnn11 8.66 / 10.19; zrc11 3.31 / 3.92; zsp10 3.48 / 4.26
- metaqa: umlp12 52.17 / 55.51; gnn12 81.71 / 87.72; zrc12 28.06 / 29.82; zsp12 29.81 / 33.35; umlp11 54.83 / 55.46; gnn11 83.81 / 87.90; zrc11 28.15 / 29.16; zsp10 30.12 / 32.23
- musique: umlp12 81.75 / 87.28; gnn12 157.63 / 165.43; zrc12 39.58 / 40.79; zsp12 48.21 / 50.58; umlp11 86.53 / 89.77; gnn11 163.35 / 172.47; zrc11 40.13 / 41.70; zsp10 49.45 / 52.14
- squad: umlp12 3.76 / 3.98; gnn12 6.28 / 7.32; zrc12 2.54 / 3.10; zsp12 2.79 / 3.39; umlp11 3.81 / 4.28; gnn11 6.41 / 7.18; zrc11 2.52 / 2.90; zsp10 2.92 / 3.92
- webqsp: umlp12 80.74 / 83.46; gnn12 124.96 / 135.69; zrc12 144.23 / 199.90; zsp12 132.19 / 177.86; umlp11 86.32 / 95.16; gnn11 131.43 / 148.61; zrc11 142.70 / 201.53; zsp10 132.10 / 182.53

Cold p50 ms by the path's position in the rotated order (first to last):

- 2wiki umlp12: 4.52, 3.88, 3.95, 4.12, 4.28, 4.05, 4.14, 3.83
- 2wiki gnn12: 6.08, 6.00, 5.55, 5.92, 6.25, 6.34, 6.17, 5.87
- 2wiki zrc12: 2.78, 2.51, 2.53, 2.43, 2.40, 2.54, 2.60, 2.50
- 2wiki zsp12: 3.10, 2.60, 2.50, 2.57, 2.43, 2.63, 2.57, 2.75
- 2wiki umlp11: 5.06, 4.46, 4.41, 4.09, 4.07, 3.99, 3.98, 4.29
- 2wiki gnn11: 6.87, 6.72, 6.43, 6.03, 6.06, 6.25, 5.86, 6.19
- 2wiki zrc11: 2.84, 2.59, 2.62, 2.69, 2.36, 2.45, 2.44, 2.34
- 2wiki zsp10: 2.93, 2.61, 2.75, 2.89, 2.67, 2.54, 2.53, 2.47
- hotpotqa umlp12: 3.74, 4.14, 4.15, 4.08, 4.22, 3.75, 3.64, 3.87
- hotpotqa gnn12: 6.41, 5.48, 6.02, 6.07, 5.90, 6.32, 5.57, 5.52
- hotpotqa zrc12: 2.79, 2.54, 2.20, 2.41, 2.55, 2.40, 2.45, 2.39
- hotpotqa zsp12: 2.98, 2.43, 2.57, 2.22, 2.48, 2.52, 2.50, 2.70
- hotpotqa umlp11: 4.56, 4.18, 3.98, 4.20, 3.56, 4.04, 4.21, 4.01
- hotpotqa gnn11: 6.31, 6.31, 5.92, 5.94, 6.24, 5.27, 5.90, 6.02
- hotpotqa zrc11: 2.66, 2.36, 2.57, 2.25, 2.29, 2.38, 2.16, 2.28
- hotpotqa zsp10: 3.05, 2.61, 2.46, 2.58, 2.38, 2.32, 2.41, 2.23
- metaqa umlp12: 49.74, 48.80, 49.91, 49.94, 49.85, 48.77, 49.83, 49.47
- metaqa gnn12: 76.54, 77.63, 76.42, 78.41, 77.64, 76.79, 75.50, 77.53
- metaqa zrc12: 26.51, 26.59, 26.74, 25.79, 26.57, 26.74, 26.25, 25.75
- metaqa zsp12: 27.22, 28.35, 28.05, 28.01, 27.73, 28.01, 28.21, 28.26
- metaqa umlp11: 51.86, 49.95, 50.79, 51.70, 51.80, 51.17, 52.12, 51.89
- metaqa gnn11: 80.82, 79.48, 77.32, 79.71, 79.25, 79.28, 77.74, 79.39
- metaqa zrc11: 26.56, 26.88, 26.44, 25.62, 26.61, 26.44, 26.83, 26.32
- metaqa zsp10: 27.93, 28.20, 28.36, 28.50, 27.18, 28.19, 28.07, 28.41
- musique umlp12: 75.38, 76.14, 75.71, 75.94, 76.48, 75.62, 75.44, 74.09
- musique gnn12: 139.30, 143.23, 146.37, 144.22, 144.52, 145.86, 143.09, 139.45
- musique zrc12: 35.74, 35.24, 36.61, 36.69, 36.40, 35.85, 35.59, 35.53
- musique zsp12: 42.47, 41.49, 40.88, 41.90, 43.27, 42.50, 42.44, 42.06
- musique umlp11: 80.89, 81.02, 80.62, 79.36, 81.58, 81.38, 80.84, 81.25
- musique gnn11: 149.29, 148.65, 148.56, 146.86, 145.73, 148.43, 150.59, 149.62
- musique zrc11: 35.97, 36.20, 36.36, 36.11, 35.55, 34.86, 35.76, 36.43
- musique zsp10: 43.24, 42.49, 42.72, 42.93, 42.03, 41.89, 40.86, 42.06
- squad umlp12: 2.73, 2.74, 2.66, 2.75, 2.75, 2.54, 2.71, 2.59
- squad gnn12: 4.57, 4.38, 4.17, 4.47, 4.30, 4.14, 4.21, 3.96
- squad zrc12: 1.55, 1.64, 1.67, 1.79, 1.60, 1.70, 1.68, 1.65
- squad zsp12: 1.76, 1.65, 1.77, 1.73, 1.92, 1.65, 1.76, 1.84
- squad umlp11: 2.71, 2.70, 2.55, 2.70, 2.71, 2.68, 2.62, 2.64
- squad gnn11: 4.73, 4.71, 4.44, 3.88, 4.46, 4.40, 4.46, 4.58
- squad zrc11: 1.58, 1.70, 1.65, 1.68, 1.62, 1.72, 1.66, 1.65
- squad zsp10: 1.87, 1.69, 1.78, 1.85, 1.70, 1.83, 1.75, 1.71
- webqsp umlp12: 69.90, 71.34, 70.70, 70.78, 67.19, 69.15, 70.20, 69.91
- webqsp gnn12: 109.16, 104.33, 106.27, 108.02, 107.52, 105.72, 108.53, 105.77
- webqsp zrc12: 87.55, 100.30, 81.20, 74.84, 87.62, 78.33, 76.22, 87.79
- webqsp zsp12: 82.12, 85.96, 93.80, 75.40, 72.38, 82.18, 76.29, 72.13
- webqsp umlp11: 73.05, 70.79, 75.34, 74.02, 71.92, 71.29, 72.90, 73.44
- webqsp gnn11: 112.20, 106.03, 104.71, 113.68, 108.76, 109.09, 107.26, 111.41
- webqsp zrc11: 87.18, 81.89, 77.10, 81.37, 90.72, 95.54, 81.64, 76.50
- webqsp zsp10: 74.08, 83.10, 76.90, 74.06, 82.86, 85.10, 93.02, 78.53

Stage p50 ms (cold), and the warm forward p50 beside:

- 2wiki umlp12: pool 0.088, compile 2.415, forward 1.580; warm forward 1.339
- 2wiki gnn12: pool 0.090, compile 2.383, pack 0.185, forward 3.340; warm forward 3.105
- 2wiki zrc12: pool 0.092, compile 1.183, edges 0.045, lean 0.569, forward 0.606; warm forward 0.415
- 2wiki zsp12: pool 0.086, compile 1.085, edges 0.043, lean 0.558, links 0.132, forward 0.678; warm forward 0.473
- 2wiki umlp11: pool 0.087, compile 2.546, forward 1.615; warm forward 1.323
- 2wiki gnn11: pool 0.091, compile 2.572, pack 0.187, forward 3.329; warm forward 2.987
- 2wiki zrc11: pool 0.091, compile 1.193, edges 0.045, lean 0.568, forward 0.602; warm forward 0.417
- 2wiki zsp10: pool 0.087, compile 1.136, edges 0.044, lean 0.566, links 0.134, forward 0.677; warm forward 0.478
- hotpotqa umlp12: pool 0.088, compile 2.257, forward 1.553; warm forward 1.269
- hotpotqa gnn12: pool 0.092, compile 2.289, pack 0.193, forward 3.270; warm forward 2.933
- hotpotqa zrc12: pool 0.093, compile 1.083, edges 0.047, lean 0.566, forward 0.606; warm forward 0.421
- hotpotqa zsp12: pool 0.089, compile 1.022, edges 0.046, lean 0.553, links 0.137, forward 0.661; warm forward 0.461
- hotpotqa umlp11: pool 0.089, compile 2.439, forward 1.565; warm forward 1.302
- hotpotqa gnn11: pool 0.092, compile 2.445, pack 0.190, forward 3.229; warm forward 3.024
- hotpotqa zrc11: pool 0.092, compile 1.062, edges 0.046, lean 0.554, forward 0.587; warm forward 0.403
- hotpotqa zsp10: pool 0.087, compile 0.989, edges 0.045, lean 0.542, links 0.135, forward 0.631; warm forward 0.458
- metaqa umlp12: pool 0.246, compile 30.099, forward 19.128; warm forward 18.920
- metaqa gnn12: pool 0.251, compile 30.103, pack 1.483, forward 45.107; warm forward 44.975
- metaqa zrc12: pool 0.254, compile 10.016, edges 0.112, lean 5.327, chains 4.414, forward 6.050; warm forward 5.698
- metaqa zsp12: pool 0.243, compile 9.927, edges 0.111, lean 5.303, chains 4.281, links 1.521, forward 6.403; warm forward 6.019
- metaqa umlp11: pool 0.247, compile 31.975, forward 19.139; warm forward 18.895
- metaqa gnn11: pool 0.251, compile 31.825, pack 1.474, forward 44.993; warm forward 44.906
- metaqa zrc11: pool 0.255, compile 10.098, edges 0.112, lean 5.337, chains 4.376, forward 6.032; warm forward 5.724
- metaqa zsp10: pool 0.243, compile 10.006, edges 0.111, lean 5.332, chains 4.258, links 1.522, forward 6.356; warm forward 6.043
- musique umlp12: pool 0.324, compile 53.794, forward 21.586; warm forward 21.260
- musique gnn12: pool 0.325, compile 53.521, pack 3.235, forward 85.837; warm forward 85.540
- musique zrc12: pool 0.328, compile 17.664, edges 0.314, lean 12.053, forward 5.504; warm forward 5.139
- musique zsp12: pool 0.321, compile 17.535, edges 0.317, lean 11.920, links 5.537, forward 6.153; warm forward 5.769
- musique umlp11: pool 0.325, compile 59.134, forward 21.468; warm forward 21.297
- musique gnn11: pool 0.327, compile 59.113, pack 3.217, forward 85.204; warm forward 85.077
- musique zrc11: pool 0.328, compile 17.731, edges 0.319, lean 11.977, forward 5.532; warm forward 5.149
- musique zsp10: pool 0.325, compile 17.670, edges 0.318, lean 12.001, links 5.594, forward 6.164; warm forward 5.744
- squad umlp12: pool 0.073, compile 1.549, forward 1.014; warm forward 0.832
- squad gnn12: pool 0.075, compile 1.551, pack 0.164, forward 2.397; warm forward 2.165
- squad zrc12: pool 0.077, compile 0.669, edges 0.043, lean 0.426, forward 0.411; warm forward 0.266
- squad zsp12: pool 0.073, compile 0.617, edges 0.041, lean 0.404, links 0.110, forward 0.443; warm forward 0.304
- squad umlp11: pool 0.073, compile 1.582, forward 0.995; warm forward 0.815
- squad gnn11: pool 0.074, compile 1.607, pack 0.163, forward 2.469; warm forward 2.212
- squad zrc11: pool 0.077, compile 0.668, edges 0.043, lean 0.409, forward 0.413; warm forward 0.262
- squad zsp10: pool 0.073, compile 0.621, edges 0.041, lean 0.403, links 0.110, forward 0.443; warm forward 0.307
- webqsp umlp12: pool 0.285, compile 48.745, forward 20.603; warm forward 20.320
- webqsp gnn12: pool 0.290, compile 48.387, pack 2.071, forward 55.822; warm forward 54.826
- webqsp zrc12: pool 0.298, compile 18.419, edges 0.157, lean 6.810, chains 41.898, forward 15.139; warm forward 14.771
- webqsp zsp12: pool 0.284, compile 18.491, edges 0.155, lean 6.710, chains 35.859, links 2.237, forward 15.409; warm forward 15.182
- webqsp umlp11: pool 0.286, compile 51.478, forward 20.516; warm forward 20.244
- webqsp gnn11: pool 0.289, compile 51.515, pack 2.055, forward 55.492; warm forward 54.907
- webqsp zrc11: pool 0.292, compile 18.493, edges 0.157, lean 6.783, chains 42.218, forward 15.175; warm forward 15.108
- webqsp zsp10: pool 0.285, compile 18.832, edges 0.154, lean 6.771, chains 35.705, links 2.216, forward 15.826; warm forward 15.326

Checks:

- 2wiki: kept 200/200; every check passed on every question; largest re-run difference 0.0e+00; seed_cos3 by pattern on 398 of 398 calls (patterns p50 63.0)
- hotpotqa: kept 200/200; every check passed on every question; largest re-run difference 0.0e+00; seed_cos3 by pattern on 396 of 396 calls (patterns p50 62.0)
- metaqa: kept 200/200; every check passed on every question; largest re-run difference 0.0e+00; seed_cos3 by pattern on 396 of 396 calls (patterns p50 96.0)
- musique: kept 200/200; every check passed on every question; largest re-run difference 0.0e+00; seed_cos3 by pattern on 398 of 398 calls (patterns p50 246.0)
- squad: kept 200/200; every check passed on every question; largest re-run difference 0.0e+00; seed_cos3 by pattern on 398 of 398 calls (patterns p50 47.0)
- webqsp: kept 200/200; every check passed on every question; largest re-run difference 0.0e+00; seed_cos3 by pattern on 398 of 398 calls (patterns p50 117.0)
