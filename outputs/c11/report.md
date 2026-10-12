Cold, batch 1, total per question, p50 ms; ratios with 2,000-resample bootstrap 95% CIs. Each model's fastest form is the faster of its new and filed paths by cold p50.

universal family:

| dataset | umlp new | umlp filed | gnn new | gnn filed | **gnn / umlp (fastest forms)** | at least 2x | warm forward gnn / umlp |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 4.71 | 7.14 (umlp9) | 6.76 | 9.29 (gnn4) | **1.44 [1.36, 1.48]** (gnn11 / umlp11) | no | 2.27 [2.20, 2.33] |
| hotpotqa | 4.35 | 6.53 (umlp9) | 6.18 | 8.30 (gnn4) | **1.42 [1.36, 1.48]** (gnn11 / umlp11) | no | 2.29 [2.20, 2.39] |
| metaqa | 57.24 | 80.29 (umlp9) | 88.42 | 106.91 (gnn5) | **1.54 [1.52, 1.57]** (gnn11 / umlp11) | no | 2.40 [2.36, 2.45] |
| musique | 83.26 | 109.57 (umlp9) | 154.13 | 178.98 (gnn4) | **1.85 [1.81, 1.88]** (gnn11 / umlp11) | no | 4.14 [4.05, 4.19] |
| squad | 2.95 | 4.25 (umlp9) | 4.61 | 6.09 (gnn4) | **1.56 [1.52, 1.62]** (gnn11 / umlp11) | no | 2.73 [2.60, 2.87] |
| webqsp | 77.79 | 106.41 (umlp9) | 117.33 | 141.14 (gnn5) | **1.51 [1.49, 1.52]** (gnn11 / umlp11) | no | 2.73 [2.70, 2.78] |

lean family:

| dataset | zrc new | zrc filed | zsp new | zsp filed | **zsp / zrc (fastest forms)** | at least 2x | warm forward zsp / zrc |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 2.72 | 4.34 (zrc4) | 2.76 | 4.80 (zsp4) | **1.01 [0.99, 1.05]** (zsp10 / zrc11) | no | 1.11 [1.09, 1.14] |
| hotpotqa | 2.41 | 3.95 (zrc4) | 2.52 | 4.27 (zsp4) | **1.04 [1.02, 1.09]** (zsp10 / zrc11) | no | 1.14 [1.10, 1.17] |
| metaqa | 29.77 | 45.52 (zrc6) | 32.18 | 47.68 (zsp6) | **1.08 [1.06, 1.10]** (zsp10 / zrc11) | no | 1.06 [1.02, 1.10] |
| musique | 37.16 | 54.36 (zrc4) | 43.71 | 61.55 (zsp4) | **1.18 [1.16, 1.19]** (zsp10 / zrc11) | no | 1.12 [1.10, 1.13] |
| squad | 1.81 | 2.61 (zrc4) | 1.90 | 3.00 (zsp4) | **1.05 [1.02, 1.08]** (zsp10 / zrc11) | no | 1.15 [1.11, 1.17] |
| webqsp | 88.66 | 111.79 (zrc6) | 84.54 | 108.30 (zsp6) | **0.95 [0.93, 0.98]** (zsp10 / zrc11) | no | 1.02 [1.00, 1.04] |

Across the families (fastest forms):

| dataset | fastest GNN | fastest GNN / universal MLP | fastest GNN / zrc | universal GNN / zrc |
| --- | --- | --- | --- | --- |
| 2wiki | zsp10 | 0.59 [0.56, 0.61] | 1.01 [0.99, 1.05] | 2.49 [2.38, 2.56] |
| hotpotqa | zsp10 | 0.58 [0.56, 0.61] | 1.04 [1.02, 1.09] | 2.56 [2.45, 2.66] |
| metaqa | zsp10 | 0.56 [0.55, 0.57] | 1.08 [1.06, 1.09] | 2.97 [2.92, 3.02] |
| musique | zsp10 | 0.52 [0.52, 0.53] | 1.18 [1.16, 1.19] | 4.15 [4.07, 4.19] |
| squad | zsp10 | 0.64 [0.62, 0.66] | 1.05 [1.02, 1.08] | 2.54 [2.45, 2.64] |
| webqsp | zsp10 | 1.09 [1.05, 1.18] | 0.95 [0.93, 0.98] | 1.32 [1.24, 1.36] |

What this stage's forms give each model (filed path / new path, cold):

| dataset | umlp9 / umlp11 | GNN filed / gnn11 | zrc filed / zrc11 | zsp filed / zsp10 |
| --- | --- | --- | --- | --- |
| 2wiki | 1.52 [1.43, 1.57] | 1.38 [1.32, 1.43] | 1.60 [1.55, 1.66] | 1.74 [1.66, 1.78] |
| hotpotqa | 1.50 [1.43, 1.56] | 1.34 [1.29, 1.41] | 1.64 [1.58, 1.71] | 1.70 [1.60, 1.77] |
| metaqa | 1.40 [1.38, 1.43] | 1.21 [1.18, 1.23] | 1.53 [1.49, 1.55] | 1.48 [1.45, 1.51] |
| musique | 1.32 [1.29, 1.33] | 1.16 [1.15, 1.18] | 1.46 [1.43, 1.48] | 1.41 [1.39, 1.43] |
| squad | 1.44 [1.41, 1.48] | 1.32 [1.25, 1.37] | 1.44 [1.40, 1.48] | 1.58 [1.52, 1.63] |
| webqsp | 1.37 [1.35, 1.39] | 1.20 [1.19, 1.22] | 1.26 [1.21, 1.29] | 1.28 [1.22, 1.31] |

Cold p95 / p99 ms:

- 2wiki: umlp11 46.52 / 137.50; gnn11 57.99 / 113.40; zrc11 53.27 / 106.28; zsp10 51.24 / 101.74; umlp9 65.18 / 137.22; gnn4 54.91 / 103.87; zrc4 47.21 / 77.37; zsp4 49.82 / 102.34
- hotpotqa: umlp11 9.37 / 20.39; gnn11 11.83 / 27.59; zrc11 6.89 / 13.94; zsp10 8.00 / 16.54; umlp9 12.49 / 24.09; gnn4 14.28 / 32.63; zrc4 7.96 / 19.05; zsp4 7.74 / 20.80
- metaqa: umlp11 100.96 / 120.08; gnn11 152.98 / 197.20; zrc11 49.42 / 64.55; zsp10 52.30 / 59.79; umlp9 137.31 / 163.13; gnn5 202.36 / 237.77; zrc6 76.26 / 98.36; zsp6 76.12 / 94.60
- musique: umlp11 112.48 / 193.82; gnn11 209.11 / 321.71; zrc11 51.31 / 63.29; zsp10 60.82 / 79.61; umlp9 144.05 / 209.15; gnn4 259.18 / 335.30; zrc4 73.14 / 91.15; zsp4 83.31 / 131.73
- squad: umlp11 4.16 / 6.44; gnn11 6.53 / 7.12; zrc11 2.51 / 3.01; zsp10 2.72 / 3.58; umlp9 5.81 / 7.76; gnn4 8.37 / 9.25; zrc4 3.54 / 4.15; zsp4 4.33 / 6.71
- webqsp: umlp11 91.12 / 132.86; gnn11 134.27 / 151.56; zrc11 154.03 / 271.99; zsp10 139.67 / 237.17; umlp9 124.21 / 171.08; gnn5 163.06 / 192.39; zrc6 174.84 / 283.97; zsp6 166.27 / 260.50

Cold p50 ms by the path's position in the rotated order (first to last):

- 2wiki umlp11: 40.72, 4.50, 4.37, 4.69, 4.85, 4.76, 4.50, 4.45
- 2wiki gnn11: 47.97, 6.22, 6.32, 6.38, 6.67, 6.78, 6.63, 6.80
- 2wiki zrc11: 46.19, 2.66, 2.72, 2.41, 2.60, 2.66, 2.69, 2.66
- 2wiki zsp10: 45.42, 2.74, 2.70, 2.71, 2.46, 2.75, 2.65, 2.76
- 2wiki umlp9: 60.18, 7.07, 7.16, 6.99, 6.86, 6.31, 6.51, 6.77
- 2wiki gnn4: 44.75, 9.66, 9.13, 8.65, 8.95, 8.38, 8.43, 9.08
- 2wiki zrc4: 41.75, 4.23, 4.30, 4.27, 4.35, 4.28, 4.23, 3.81
- 2wiki zsp4: 40.33, 4.61, 4.84, 4.81, 4.73, 4.63, 4.59, 4.47
- hotpotqa umlp11: 6.36, 4.11, 3.96, 4.26, 4.43, 4.15, 4.19, 4.28
- hotpotqa gnn11: 9.69, 5.70, 5.59, 5.53, 6.18, 6.56, 5.48, 5.97
- hotpotqa zrc11: 4.89, 2.46, 2.30, 2.20, 2.26, 2.39, 2.47, 2.38
- hotpotqa zsp10: 5.99, 2.64, 2.52, 2.36, 2.37, 2.29, 2.37, 2.57
- hotpotqa umlp9: 9.55, 6.57, 6.56, 6.40, 6.22, 6.09, 6.01, 6.12
- hotpotqa gnn4: 11.18, 8.47, 8.03, 8.21, 7.97, 8.10, 8.08, 7.58
- hotpotqa zrc4: 6.68, 3.86, 3.91, 3.63, 3.77, 3.97, 3.63, 3.95
- hotpotqa zsp4: 7.37, 4.27, 4.20, 4.60, 3.97, 4.10, 4.11, 3.99
- metaqa umlp11: 59.52, 58.31, 58.33, 56.98, 55.84, 55.14, 57.19, 57.56
- metaqa gnn11: 92.14, 91.48, 91.35, 96.50, 89.68, 85.70, 83.95, 86.79
- metaqa zrc11: 27.97, 29.78, 31.05, 32.55, 30.56, 31.08, 29.41, 28.04
- metaqa zsp10: 30.51, 30.69, 32.16, 32.17, 33.74, 33.15, 33.03, 32.06
- metaqa umlp9: 77.56, 78.79, 76.15, 79.63, 82.21, 84.71, 85.41, 80.91
- metaqa gnn5: 112.78, 106.17, 105.15, 103.89, 106.94, 109.28, 120.97, 107.12
- metaqa zrc6: 48.30, 45.35, 44.29, 44.92, 43.18, 44.47, 46.48, 46.26
- metaqa zsp6: 48.61, 51.57, 49.02, 46.16, 46.26, 46.13, 48.01, 49.84
- musique umlp11: 83.71, 82.63, 84.27, 82.56, 83.56, 82.65, 85.02, 81.90
- musique gnn11: 149.14, 155.98, 157.50, 153.57, 155.50, 151.97, 149.97, 156.13
- musique zrc11: 37.14, 36.33, 38.10, 37.36, 37.38, 37.02, 36.66, 37.38
- musique zsp10: 44.48, 44.01, 42.56, 43.88, 43.70, 42.92, 45.55, 42.73
- musique umlp9: 108.57, 112.21, 107.70, 107.28, 109.84, 109.61, 107.72, 111.94
- musique gnn4: 185.39, 182.03, 181.97, 184.72, 174.56, 177.32, 182.40, 175.87
- musique zrc4: 55.72, 54.56, 53.15, 56.35, 53.03, 53.21, 54.70, 53.92
- musique zsp4: 62.73, 62.98, 62.03, 61.58, 61.38, 60.84, 60.41, 61.52
- squad umlp11: 3.08, 3.11, 2.95, 3.04, 2.99, 2.83, 2.75, 2.90
- squad gnn11: 4.95, 4.62, 4.72, 4.68, 4.93, 4.90, 4.25, 4.09
- squad zrc11: 1.77, 1.81, 1.79, 1.89, 1.75, 1.84, 1.91, 1.74
- squad zsp10: 1.98, 1.70, 1.86, 1.83, 1.96, 1.82, 1.89, 1.99
- squad umlp9: 4.49, 4.18, 4.06, 4.13, 4.23, 4.32, 4.33, 4.35
- squad gnn4: 6.55, 6.70, 5.79, 5.61, 6.21, 6.00, 6.66, 5.81
- squad zrc4: 2.73, 2.71, 2.67, 2.42, 2.53, 2.55, 2.58, 2.66
- squad zsp4: 3.49, 3.09, 3.11, 3.04, 2.80, 2.74, 2.85, 2.97
- webqsp umlp11: 80.64, 77.11, 77.45, 79.24, 75.30, 76.73, 77.86, 77.68
- webqsp gnn11: 118.33, 119.35, 114.76, 116.86, 119.69, 113.61, 118.09, 117.26
- webqsp zrc11: 107.43, 107.47, 85.97, 85.25, 90.03, 88.52, 81.33, 92.96
- webqsp zsp10: 92.44, 102.88, 100.48, 81.01, 80.73, 84.55, 83.39, 76.39
- webqsp umlp9: 107.25, 105.58, 107.38, 105.22, 107.91, 105.45, 104.72, 107.16
- webqsp gnn5: 153.50, 137.47, 139.35, 142.88, 143.41, 140.36, 139.03, 139.85
- webqsp zrc6: 126.07, 111.08, 103.61, 114.62, 122.29, 126.13, 108.94, 106.75
- webqsp zsp6: 103.81, 108.17, 108.35, 102.00, 112.38, 112.43, 122.41, 106.25

Stage p50 ms (cold), and the warm forward p50 beside:

- 2wiki umlp11: pool 0.129, compile 2.856, forward 1.718; warm forward 1.414
- 2wiki gnn11: pool 0.093, compile 2.806, pack 0.198, forward 3.480; warm forward 3.215
- 2wiki zrc11: pool 0.093, compile 1.266, edges 0.046, lean 0.595, forward 0.661; warm forward 0.451
- 2wiki zsp10: pool 0.089, compile 1.188, edges 0.045, lean 0.578, links 0.135, forward 0.721; warm forward 0.500
- 2wiki umlp9: pool 0.512, read 0.818, compile 2.667, pack 0.070, forward 2.932; warm forward 2.382
- 2wiki gnn4: pool 0.533, read 0.778, compile 2.697, pack 0.211, forward 4.779; warm forward 4.305
- 2wiki zrc4: pool 0.539, read 0.776, compile 1.113, edges 0.046, lean 0.630, forward 1.144; warm forward 0.828
- 2wiki zsp4: pool 0.526, read 0.772, compile 1.083, edges 0.045, lean 0.616, links 0.134, forward 1.491; warm forward 1.117
- hotpotqa umlp11: pool 0.117, compile 2.656, forward 1.520; warm forward 1.283
- hotpotqa gnn11: pool 0.090, compile 2.518, pack 0.188, forward 3.195; warm forward 2.933
- hotpotqa zrc11: pool 0.091, compile 1.079, edges 0.046, lean 0.563, forward 0.598; warm forward 0.405
- hotpotqa zsp10: pool 0.085, compile 0.999, edges 0.045, lean 0.556, links 0.134, forward 0.662; warm forward 0.460
- hotpotqa umlp9: pool 0.501, read 0.681, compile 2.390, pack 0.061, forward 2.715; warm forward 2.137
- hotpotqa gnn4: pool 0.519, read 0.667, compile 2.377, pack 0.202, forward 4.381; warm forward 3.993
- hotpotqa zrc4: pool 0.528, read 0.674, compile 0.955, edges 0.045, lean 0.598, forward 1.060; warm forward 0.769
- hotpotqa zsp4: pool 0.507, read 0.660, compile 0.880, edges 0.044, lean 0.584, links 0.131, forward 1.389; warm forward 1.069
- metaqa umlp11: pool 0.270, compile 35.980, forward 20.788; warm forward 20.764
- metaqa gnn11: pool 0.271, compile 35.978, pack 1.738, forward 50.988; warm forward 49.830
- metaqa zrc11: pool 0.276, compile 11.711, edges 0.128, lean 5.940, chains 4.992, forward 6.860; warm forward 6.620
- metaqa zsp10: pool 0.264, compile 11.684, edges 0.129, lean 6.084, chains 4.895, links 1.647, forward 7.419; warm forward 7.038
- metaqa umlp9: pool 2.180, read 8.306, compile 35.174, pack 0.962, forward 33.843; warm forward 29.221
- metaqa gnn5: pool 2.258, read 8.428, compile 35.383, pack 1.715, forward 59.626; warm forward 60.049
- metaqa zrc6: pool 2.216, read 8.375, compile 10.268, edges 0.124, lean 8.942, chains 4.935, forward 9.964; warm forward 9.516
- metaqa zsp6: pool 2.236, read 8.635, compile 10.403, edges 0.125, lean 8.813, chains 4.781, links 1.633, forward 10.845; warm forward 10.407
- musique umlp11: pool 0.329, compile 61.276, forward 21.724; warm forward 21.445
- musique gnn11: pool 0.334, compile 61.044, pack 3.639, forward 88.625; warm forward 88.710
- musique zrc11: pool 0.335, compile 18.405, edges 0.336, lean 12.509, forward 5.642; warm forward 5.230
- musique zsp10: pool 0.329, compile 18.368, edges 0.338, lean 12.531, links 5.763, forward 6.269; warm forward 5.857
- musique umlp9: pool 3.212, read 8.766, compile 60.211, pack 1.044, forward 36.299; warm forward 30.497
- musique gnn4: pool 3.296, read 8.912, compile 60.086, pack 3.715, forward 102.293; warm forward 105.549
- musique zrc4: pool 3.335, read 8.980, compile 17.211, edges 0.345, lean 15.484, forward 8.601; warm forward 8.056
- musique zsp4: pool 3.276, read 9.168, compile 17.110, edges 0.340, lean 15.561, links 5.911, forward 9.750; warm forward 9.271
- squad umlp11: pool 0.081, compile 1.761, forward 1.083; warm forward 0.862
- squad gnn11: pool 0.080, compile 1.746, pack 0.178, forward 2.614; warm forward 2.353
- squad zrc11: pool 0.082, compile 0.748, edges 0.047, lean 0.454, forward 0.453; warm forward 0.287
- squad zsp10: pool 0.076, compile 0.675, edges 0.043, lean 0.431, links 0.117, forward 0.478; warm forward 0.329
- squad umlp9: pool 0.421, read 0.122, compile 1.599, pack 0.046, forward 2.022; warm forward 1.654
- squad gnn4: pool 0.449, read 0.124, compile 1.610, pack 0.182, forward 3.665; warm forward 3.329
- squad zrc4: pool 0.463, read 0.124, compile 0.637, edges 0.045, lean 0.473, forward 0.850; warm forward 0.605
- squad zsp4: pool 0.432, read 0.122, compile 0.618, edges 0.043, lean 0.458, links 0.110, forward 1.150; warm forward 0.890
- webqsp umlp11: pool 0.322, compile 55.351, forward 22.012; warm forward 21.373
- webqsp gnn11: pool 0.305, compile 55.125, pack 2.233, forward 59.376; warm forward 58.386
- webqsp zrc11: pool 0.310, compile 19.966, edges 0.162, lean 7.239, chains 44.538, forward 16.304; warm forward 15.988
- webqsp zsp10: pool 0.302, compile 20.164, edges 0.160, lean 7.098, chains 37.299, links 2.314, forward 16.731; warm forward 16.237
- webqsp umlp9: pool 6.056, read 9.986, compile 53.057, pack 1.127, forward 35.461; warm forward 30.078
- webqsp gnn5: pool 6.050, read 9.950, compile 53.679, pack 2.266, forward 69.187; warm forward 70.314
- webqsp zrc6: pool 6.012, read 10.120, compile 18.370, edges 0.158, lean 10.630, chains 45.618, forward 20.250; warm forward 19.699
- webqsp zsp6: pool 5.960, read 10.027, compile 18.229, edges 0.160, lean 10.525, chains 37.544, links 2.323, forward 20.872; warm forward 20.204

Checks:

- 2wiki: kept 200/200; every check passed on every question; largest difference from the module: gnn11 3.3e-06, gnn4 1.9e-06, umlp11 5.7e-06, umlp9 0.0e+00, zrc11 1.0e-05, zrc4 7.6e-06, zsp10 5.7e-06, zsp4 5.2e-06; largest re-run difference 0.0e+00
- hotpotqa: kept 200/200; every check passed on every question; largest difference from the module: gnn11 2.9e-06, gnn4 2.9e-06, umlp11 5.7e-06, umlp9 0.0e+00, zrc11 7.6e-06, zrc4 7.6e-06, zsp10 5.7e-06, zsp4 5.7e-06; largest re-run difference 0.0e+00
- metaqa: kept 200/200; every check passed on every question; largest difference from the module: gnn11 6.4e-06, gnn5 4.8e-06, umlp11 7.6e-06, umlp9 0.0e+00, zrc11 8.1e-05, zrc6 8.1e-05, zsp10 6.3e-05, zsp6 6.4e-05; largest re-run difference 0.0e+00
- musique: kept 200/200; every check passed on every question; largest difference from the module: gnn11 4.8e-06, gnn4 3.8e-06, umlp11 7.6e-06, umlp9 0.0e+00, zrc11 8.5e-05, zrc4 8.5e-05, zsp10 5.6e-05, zsp4 5.6e-05; largest re-run difference 0.0e+00
- squad: kept 200/200; every check passed on every question; largest difference from the module: gnn11 2.9e-06, gnn4 1.9e-06, umlp11 3.8e-06, umlp9 0.0e+00, zrc11 3.3e-06, zrc4 3.6e-06, zsp10 3.2e-06, zsp4 2.6e-06; largest re-run difference 0.0e+00
- webqsp: kept 200/200; every check passed on every question; largest difference from the module: gnn11 6.7e-06, gnn5 5.7e-06, umlp11 1.1e-05, umlp9 0.0e+00, zrc11 9.2e-05, zrc6 8.8e-05, zsp10 7.8e-05, zsp6 7.7e-05; largest re-run difference 0.0e+00
