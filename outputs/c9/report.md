Cold, batch 1, total per question, p50 ms; ratios GNN / MLP within each family (bootstrap 95% CI):

| dataset | universal MLP | universal GNN | **universal GNN / MLP** | lean MLP zrc | lean GNN zsp | **lean zsp / zrc** | universal GNN / zrc | universal MLP / zrc | warm forward universal GNN / MLP | warm forward zsp / zrc |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 2wiki | 12.0 | 15.2 | **1.27 [1.22, 1.32]** | 7.8 | 8.2 | **1.04 [0.97, 1.13]** | 1.94 [1.86, 2.09] | 1.53 [1.46, 1.63] | 2.03 [1.86, 2.19] | 1.35 [1.28, 1.41] |
| hotpotqa | 10.9 | 14.1 | **1.29 [1.24, 1.34]** | 6.5 | 7.8 | **1.21 [1.11, 1.28]** | 2.17 [2.06, 2.28] | 1.69 [1.61, 1.75] | 1.78 [1.66, 1.96] | 1.42 [1.33, 1.52] |
| metaqa | 125.3 | 175.6 | **1.40 [1.36, 1.43]** | 68.2 | 71.4 | **1.05 [1.03, 1.07]** | 2.57 [2.51, 2.64] | 1.84 [1.80, 1.89] | 2.10 [2.02, 2.18] | 1.06 [1.04, 1.09] |
| musique | 195.9 | 322.6 | **1.65 [1.60, 1.70]** | 93.3 | 105.8 | **1.13 [1.10, 1.16]** | 3.46 [3.35, 3.56] | 2.10 [2.04, 2.14] | 3.20 [3.09, 3.36] | 1.16 [1.13, 1.21] |
| squad | 7.8 | 10.9 | **1.40 [1.32, 1.53]** | 4.4 | 5.0 | **1.16 [1.06, 1.27]** | 2.51 [2.35, 2.67] | 1.80 [1.62, 1.93] | 2.21 [2.03, 2.38] | 1.50 [1.42, 1.58] |
| webqsp | 152.0 | 215.4 | **1.42 [1.33, 1.49]** | 163.0 | 153.8 | **0.94 [0.91, 0.98]** | 1.32 [1.24, 1.41] | 0.93 [0.89, 1.00] | 2.49 [2.35, 2.65] | 1.00 [0.96, 1.08] |

Stage p50 ms (cold), and the warm forward p50 beside:

- 2wiki gnn4: pool 0.82, read 1.20, compile 4.28, pack 0.33, forward 7.81; warm forward 7.13
- 2wiki umlp9: pool 0.85, read 1.24, compile 4.44, pack 0.12, forward 4.54; warm forward 3.51
- 2wiki zsp4: pool 0.83, read 1.16, compile 1.74, edges 0.07, lean 0.93, links 0.21, forward 2.38; warm forward 1.68
- 2wiki zrc4: pool 0.84, read 1.22, compile 1.86, edges 0.07, lean 0.98, forward 1.83; warm forward 1.25
- hotpotqa gnn4: pool 0.82, read 1.04, compile 3.92, pack 0.35, forward 7.72; warm forward 6.74
- hotpotqa umlp9: pool 0.83, read 1.08, compile 3.98, pack 0.12, forward 4.49; warm forward 3.79
- hotpotqa zsp4: pool 0.80, read 1.08, compile 1.56, edges 0.07, lean 0.98, links 0.23, forward 2.48; warm forward 1.75
- hotpotqa zrc4: pool 0.84, read 1.04, compile 1.55, edges 0.07, lean 0.94, forward 1.78; warm forward 1.23
- metaqa gnn5: pool 3.15, read 12.09, compile 53.87, pack 2.51, forward 101.59; warm forward 103.18
- metaqa umlp9: pool 3.19, read 12.17, compile 54.07, pack 1.32, forward 52.68; warm forward 49.12
- metaqa zsp6: pool 3.26, read 11.67, compile 15.75, edges 0.17, lean 13.21, chains 7.07, links 2.22, forward 16.57; warm forward 16.03
- metaqa zrc6: pool 3.19, read 11.60, compile 15.57, edges 0.18, lean 13.26, chains 7.10, forward 15.67; warm forward 15.18
- musique gnn4: pool 5.60, read 13.71, compile 106.54, pack 6.12, forward 190.72; warm forward 187.24
- musique umlp9: pool 5.83, read 13.38, compile 107.04, pack 1.59, forward 66.87; warm forward 58.60
- musique zsp4: pool 5.98, read 12.96, compile 30.49, edges 0.55, lean 25.66, links 9.57, forward 18.04; warm forward 17.24
- musique zrc4: pool 5.39, read 13.40, compile 29.76, edges 0.55, lean 25.12, forward 18.72; warm forward 14.83
- squad gnn4: pool 0.70, read 0.18, compile 2.62, pack 0.30, forward 6.42; warm forward 5.86
- squad umlp9: pool 0.72, read 0.18, compile 2.67, pack 0.09, forward 3.26; warm forward 2.65
- squad zsp4: pool 0.68, read 0.18, compile 0.99, edges 0.07, lean 0.75, links 0.19, forward 1.87; warm forward 1.36
- squad zrc4: pool 0.73, read 0.18, compile 0.99, edges 0.07, lean 0.81, forward 1.37; warm forward 0.90
- webqsp gnn5: pool 8.07, read 15.15, compile 77.54, pack 3.01, forward 109.59; warm forward 107.37
- webqsp umlp9: pool 7.99, read 15.07, compile 74.03, pack 1.59, forward 48.08; warm forward 43.10
- webqsp zsp6: pool 8.21, read 14.86, compile 24.84, edges 0.20, lean 14.06, chains 52.17, links 3.11, forward 31.92; warm forward 28.54
- webqsp zrc6: pool 7.91, read 15.21, compile 25.45, edges 0.21, lean 14.48, chains 61.65, forward 30.81; warm forward 28.50

Checks:

- 2wiki: kept 200/200; {'questions': 200, 'kept': 200, 'zrc4_compile_bits': 200, 'zrc4_scores_equal': 200, 'zsp4_compile_bits': 200, 'zsp4_scores_equal': 200, 'gnn4_compile_bits': 200, 'gnn4_scores_equal': 200, 'gnn4_top5_look': 200, 'pool': 200, 'umlp9_pool': 200, 'umlp9_within_tol': 200, 'umlp9_top5_same': 200, 'umlp9_top5_look': 200, 'gnn4_vs_look_max': 3.814697265625e-06, 'umlp9_diff_max': 0.0, 'umlp9_vs_look_max': 5.7220458984375e-06}
- hotpotqa: kept 200/200; {'questions': 200, 'kept': 200, 'zrc4_compile_bits': 200, 'zrc4_scores_equal': 200, 'zsp4_compile_bits': 200, 'zsp4_scores_equal': 200, 'gnn4_compile_bits': 200, 'gnn4_scores_equal': 200, 'gnn4_top5_look': 200, 'pool': 200, 'umlp9_pool': 200, 'umlp9_within_tol': 200, 'umlp9_top5_same': 200, 'umlp9_top5_look': 200, 'gnn4_vs_look_max': 3.814697265625e-06, 'umlp9_diff_max': 0.0, 'umlp9_vs_look_max': 5.7220458984375e-06}
- metaqa: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'edges': 200, 'e_rel': 200, 'zrc5_entries': 200, 'zsp5_entries': 200, 'zrc5_within_tol': 200, 'zrc5_top5_same': 200, 'zsp5_within_tol': 200, 'zsp5_top5_same': 200, 'gnn5_top5_look': 200, 'zrc6_entries_bits': 200, 'zsp6_entries_bits': 200, 'zrc6_within_tol': 200, 'zrc6_top5_same': 200, 'zsp6_within_tol': 200, 'zsp6_top5_same': 200, 'umlp9_pool': 200, 'umlp9_within_tol': 200, 'umlp9_top5_same': 200, 'umlp9_top5_look': 200, 'gnn5_vs_look_max': 1.4781951904296875e-05, 'umlp9_diff_max': 0.0, 'umlp9_vs_look_max': 1.1444091796875e-05, 'zrc5_diff_max': 8.058547973632812e-05, 'zrc6_diff_max': 8.058547973632812e-05, 'zsp5_diff_max': 6.365776062011719e-05, 'zsp6_diff_max': 6.365776062011719e-05}
- musique: kept 200/200; {'questions': 200, 'kept': 200, 'zrc4_compile_bits': 200, 'zrc4_scores_equal': 200, 'zsp4_compile_bits': 200, 'zsp4_scores_equal': 200, 'gnn4_compile_bits': 200, 'gnn4_scores_equal': 200, 'gnn4_top5_look': 200, 'pool': 200, 'umlp9_pool': 200, 'umlp9_within_tol': 200, 'umlp9_top5_same': 200, 'umlp9_top5_look': 200, 'gnn4_vs_look_max': 4.76837158203125e-06, 'umlp9_diff_max': 0.0, 'umlp9_vs_look_max': 7.62939453125e-06}
- squad: kept 200/200; {'questions': 200, 'kept': 200, 'zrc4_compile_bits': 200, 'zrc4_scores_equal': 200, 'zsp4_compile_bits': 200, 'zsp4_scores_equal': 200, 'gnn4_compile_bits': 200, 'gnn4_scores_equal': 200, 'gnn4_top5_look': 200, 'pool': 200, 'umlp9_pool': 200, 'umlp9_within_tol': 200, 'umlp9_top5_same': 200, 'umlp9_top5_look': 200, 'gnn4_vs_look_max': 2.86102294921875e-06, 'umlp9_diff_max': 0.0, 'umlp9_vs_look_max': 5.7220458984375e-06}
- webqsp: kept 200/200; {'questions': 200, 'kept': 200, 'pool': 200, 'seeds': 200, 'edges': 200, 'e_rel': 200, 'zrc5_entries': 200, 'zsp5_entries': 200, 'zrc5_within_tol': 200, 'zrc5_top5_same': 200, 'zsp5_within_tol': 200, 'zsp5_top5_same': 200, 'gnn5_top5_look': 200, 'zrc6_entries_bits': 200, 'zsp6_entries_bits': 200, 'zrc6_within_tol': 200, 'zrc6_top5_same': 200, 'zsp6_within_tol': 200, 'zsp6_top5_same': 200, 'umlp9_pool': 200, 'umlp9_within_tol': 200, 'umlp9_top5_same': 200, 'umlp9_top5_look': 200, 'gnn5_vs_look_max': 8.106231689453125e-06, 'umlp9_diff_max': 0.0, 'umlp9_vs_look_max': 1.1444091796875e-05, 'zrc5_diff_max': 8.7738037109375e-05, 'zrc6_diff_max': 8.7738037109375e-05, 'zsp5_diff_max': 7.724761962890625e-05, 'zsp6_diff_max': 7.724761962890625e-05}
