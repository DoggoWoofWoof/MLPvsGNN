Cold, batch 1, total per question, p50 ms; ratios with bootstrap 95% CIs. The headline is the faster GNN form (zsp10 or zsp's filed path) over zn10.

| dataset | zn10 | zn4 | zsp10 | zsp filed (path) | **fastest GNN / zn10** | at least 2x | zsp10 / zn10 | zsp filed / zn10 | zn4 / zn10 | zsp filed / zsp10 | zsp filed / zn4 | warm forward zsp10 / zn10 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 1.11 | 2.69 | 5.79 | 11.68 (zsp4) | **5.22 [3.50, 13.57]** (zsp10) | yes (CI above 2) | 5.22 [3.58, 13.53] | 10.53 [6.09, 15.21] | 2.43 [2.36, 2.51] | 2.02 [0.45, 4.22] | 4.34 [2.48, 6.27] | 1.43 [1.38, 1.49] |
| hotpotqa | 1.05 | 2.52 | 2.78 | 4.73 (zsp4) | **2.65 [2.58, 2.75]** (zsp10) | yes (CI above 2) | 2.65 [2.59, 2.76] | 4.50 [4.37, 4.76] | 2.40 [2.34, 2.49] | 1.70 [1.64, 1.79] | 1.88 [1.81, 1.99] | 1.44 [1.40, 1.49] |
| metaqa | 7.78 | 19.98 | 30.71 | 44.87 (zsp6) | **3.95 [3.88, 4.01]** (zsp10) | yes (CI above 2) | 3.95 [3.89, 4.01] | 5.77 [5.69, 5.85] | 2.57 [2.53, 2.61] | 1.46 [1.44, 1.48] | 2.25 [2.21, 2.27] | 1.72 [1.69, 1.76] |
| musique | 9.05 | 23.59 | 48.11 | 65.26 (zsp4) | **5.31 [5.20, 5.43]** (zsp10) | yes (CI above 2) | 5.31 [5.20, 5.43] | 7.21 [7.08, 7.38] | 2.61 [2.55, 2.66] | 1.36 [1.34, 1.38] | 2.77 [2.72, 2.83] | 1.55 [1.51, 1.59] |
| squad | 0.75 | 1.54 | 2.02 | 3.18 (zsp4) | **2.71 [2.62, 2.86]** (zsp10) | yes (CI above 2) | 2.71 [2.62, 2.86] | 4.26 [4.12, 4.41] | 2.07 [2.02, 2.16] | 1.57 [1.50, 1.62] | 2.06 [1.98, 2.12] | 1.41 [1.38, 1.44] |
| webqsp | 8.86 | 27.33 | 85.07 | 106.35 (zsp6) | **9.61 [9.05, 10.08]** (zsp10) | yes (CI above 2) | 9.61 [9.08, 10.08] | 12.01 [11.24, 12.66] | 3.09 [2.97, 3.16] | 1.25 [1.21, 1.28] | 3.89 [3.69, 4.08] | 4.12 [3.99, 4.26] |

Cold p95 / p99 ms:

- 2wiki: zn10 2.02 / 2.88; zsp10 78.25 / 150.29; zn4 4.46 / 6.20; zsp4 65.81 / 125.57
- hotpotqa: zn10 1.98 / 2.51; zsp10 4.49 / 5.45; zn4 4.07 / 5.29; zsp4 7.02 / 8.59
- metaqa: zn10 10.22 / 12.82; zsp10 44.00 / 59.77; zn4 27.15 / 37.36; zsp6 61.17 / 74.49
- musique: zn10 16.33 / 18.95; zsp10 81.60 / 102.42; zn4 39.49 / 50.41; zsp4 116.54 / 142.00
- squad: zn10 1.13 / 1.40; zsp10 3.16 / 3.79; zn4 2.26 / 2.76; zsp4 4.41 / 6.49
- webqsp: zn10 13.58 / 26.03; zsp10 139.34 / 225.16; zn4 34.32 / 37.19; zsp6 159.02 / 221.68

Stage p50 ms (cold), and the warm forward p50 beside:

- 2wiki zn10: pool 0.144, read 0.132, retrieval 0.115, seed 0.165, forward 0.526; warm forward 0.365
- 2wiki zsp10: pool 0.092, compile 4.101, edges 0.052, lean 0.612, links 0.140, forward 0.845; warm forward 0.521
- 2wiki zn4: pool 0.550, read 0.813, retrieval 0.099, seed 0.196, forward 1.000; warm forward 0.716
- 2wiki zsp4: pool 0.527, read 0.822, compile 4.309, edges 0.051, lean 0.662, links 0.142, forward 1.673; warm forward 1.201
- hotpotqa zn10: pool 0.131, read 0.132, retrieval 0.110, seed 0.164, forward 0.495; warm forward 0.329
- hotpotqa zsp10: pool 0.088, compile 1.177, edges 0.048, lean 0.573, links 0.134, forward 0.697; warm forward 0.474
- hotpotqa zn4: pool 0.534, read 0.701, retrieval 0.092, seed 0.196, forward 0.952; warm forward 0.680
- hotpotqa zsp4: pool 0.513, read 0.706, compile 1.049, edges 0.047, lean 0.620, links 0.133, forward 1.533; warm forward 1.144
- metaqa zn10: pool 0.264, read 1.695, retrieval 0.701, seed 0.878, forward 4.154; warm forward 3.869
- metaqa zsp10: pool 0.268, compile 11.094, edges 0.129, lean 5.743, chains 4.516, links 1.588, forward 7.062; warm forward 6.663
- metaqa zn4: pool 2.051, read 7.461, retrieval 0.754, seed 3.363, forward 6.209; warm forward 5.807
- metaqa zsp6: pool 2.112, read 7.438, compile 9.894, edges 0.128, lean 8.286, chains 4.607, links 1.600, forward 10.408; warm forward 10.023
- musique zn10: pool 0.354, read 1.913, retrieval 0.790, seed 1.025, forward 4.491; warm forward 4.185
- musique zsp10: pool 0.366, compile 20.326, edges 0.381, lean 13.674, links 6.145, forward 7.074; warm forward 6.491
- musique zn4: pool 3.550, read 8.251, retrieval 0.819, seed 3.645, forward 6.887; warm forward 6.412
- musique zsp4: pool 3.513, read 8.433, compile 18.865, edges 0.384, lean 16.430, links 6.280, forward 11.229; warm forward 10.783
- squad zn10: pool 0.083, read 0.063, retrieval 0.081, seed 0.134, forward 0.372; warm forward 0.239
- squad zsp10: pool 0.079, compile 0.754, edges 0.047, lean 0.443, links 0.119, forward 0.526; warm forward 0.337
- squad zn4: pool 0.443, read 0.129, retrieval 0.064, seed 0.147, forward 0.767; warm forward 0.549
- squad zsp4: pool 0.430, read 0.129, compile 0.668, edges 0.045, lean 0.463, links 0.115, forward 1.246; warm forward 0.908
- webqsp zn10: pool 0.313, read 2.002, retrieval 0.766, seed 1.056, forward 4.334; warm forward 3.908
- webqsp zsp10: pool 0.315, compile 20.563, edges 0.159, lean 7.011, chains 36.345, links 2.304, forward 16.472; warm forward 16.096
- webqsp zn4: pool 6.043, read 9.914, retrieval 0.699, seed 4.016, forward 6.316; warm forward 5.835
- webqsp zsp6: pool 5.991, read 9.863, compile 18.612, edges 0.157, lean 10.147, chains 37.498, links 2.327, forward 20.332; warm forward 20.133

Checks:

- 2wiki: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'read_bits': 200, 'P_bits': 200, 'zn_rows_bits': 200, 'zsp10_rows_bits': 200, 'zsp10_compile_bits': 200, 'zsp10_links_bits': 200, 'zn10_within_tol': 200, 'zn10_top5_same': 200, 'zn4_within_tol': 200, 'zn4_top5_same': 200, 'zsp10_within_tol': 200, 'zsp10_top5_same': 200, 'zsp4_within_tol': 200, 'zsp4_top5_same': 200, 'zn10_diff_max': 6.67572021484375e-06, 'zn4_diff_max': 6.67572021484375e-06, 'zsp10_diff_max': 5.7220458984375e-06, 'zsp4_diff_max': 5.245208740234375e-06}
- hotpotqa: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'read_bits': 200, 'P_bits': 200, 'zn_rows_bits': 200, 'zsp10_rows_bits': 200, 'zsp10_compile_bits': 200, 'zsp10_links_bits': 200, 'zn10_within_tol': 200, 'zn10_top5_same': 200, 'zn4_within_tol': 200, 'zn4_top5_same': 200, 'zsp10_within_tol': 200, 'zsp10_top5_same': 200, 'zsp4_within_tol': 200, 'zsp4_top5_same': 200, 'zn10_diff_max': 5.7220458984375e-06, 'zn4_diff_max': 5.7220458984375e-06, 'zsp10_diff_max': 5.7220458984375e-06, 'zsp4_diff_max': 5.7220458984375e-06}
- metaqa: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'read_bits': 200, 'P_bits': 200, 'zn_rows_bits': 200, 'zsp10_rows_bits': 200, 'zsp10_compile_bits': 200, 'zsp10_links_bits': 200, 'zn10_within_tol': 200, 'zn10_top5_same': 200, 'zn4_within_tol': 200, 'zn4_top5_same': 200, 'zsp10_within_tol': 200, 'zsp10_top5_same': 200, 'zsp6_within_tol': 200, 'zsp6_top5_same': 200, 'zsp10_entries_bits': 200, 'zsp10_entries_filed': 200, 'zsp10_e_rel': 200, 'zn10_diff_max': 6.198883056640625e-06, 'zn4_diff_max': 6.67572021484375e-06, 'zsp10_diff_max': 6.341934204101562e-05, 'zsp6_diff_max': 6.365776062011719e-05}
- musique: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'read_bits': 200, 'P_bits': 200, 'zn_rows_bits': 200, 'zsp10_rows_bits': 200, 'zsp10_compile_bits': 200, 'zsp10_links_bits': 200, 'zn10_within_tol': 200, 'zn10_top5_same': 200, 'zn4_within_tol': 200, 'zn4_top5_same': 200, 'zsp10_within_tol': 200, 'zsp10_top5_same': 200, 'zsp4_within_tol': 200, 'zsp4_top5_same': 200, 'zn10_diff_max': 9.298324584960938e-06, 'zn4_diff_max': 1.0728836059570312e-05, 'zsp10_diff_max': 5.5789947509765625e-05, 'zsp4_diff_max': 5.555152893066406e-05}
- squad: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'read_bits': 200, 'P_bits': 200, 'zn_rows_bits': 200, 'zsp10_rows_bits': 200, 'zsp10_compile_bits': 200, 'zsp10_links_bits': 200, 'zn10_within_tol': 200, 'zn10_top5_same': 200, 'zn4_within_tol': 200, 'zn4_top5_same': 200, 'zsp10_within_tol': 200, 'zsp10_top5_same': 200, 'zsp4_within_tol': 200, 'zsp4_top5_same': 200, 'zn10_diff_max': 3.5762786865234375e-06, 'zn4_diff_max': 2.86102294921875e-06, 'zsp10_diff_max': 3.2186508178710938e-06, 'zsp4_diff_max': 2.562999725341797e-06}
- webqsp: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'read_bits': 200, 'P_bits': 200, 'zn_rows_bits': 200, 'zsp10_rows_bits': 200, 'zsp10_compile_bits': 200, 'zsp10_links_bits': 200, 'zn10_within_tol': 200, 'zn10_top5_same': 200, 'zn4_within_tol': 200, 'zn4_top5_same': 200, 'zsp10_within_tol': 200, 'zsp10_top5_same': 200, 'zsp6_within_tol': 200, 'zsp6_top5_same': 200, 'zsp10_entries_bits': 200, 'zsp10_entries_filed': 200, 'zsp10_e_rel': 200, 'zn10_diff_max': 2.86102294921875e-05, 'zn4_diff_max': 2.765655517578125e-05, 'zsp10_diff_max': 7.82012939453125e-05, 'zsp6_diff_max': 7.724761962890625e-05}
