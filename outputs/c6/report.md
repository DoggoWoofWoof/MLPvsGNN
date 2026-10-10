Cold, batch 1, total per question, p50 ms:

| dataset | zrc6 | zsp6 | gnn5 | **gnn5 / zrc6** | gnn5 / zsp6 | gnn5 / zrc5 (C5, same run) | zrc5 / zrc6 | zsp5 / zsp6 | chains p50 zrc5 -> zrc6 | forward p50 zrc5 -> zrc6 | warm forward gnn5 / zrc6 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 40.5 | 42.5 | 97.8 | **2.42 [2.40, 2.44]** | 2.30 [2.29, 2.32] | 2.31 [2.29, 2.34] | 1.05 [1.04, 1.06] | 1.04 [1.04, 1.05] | 5.64 -> 4.51 | 9.71 -> 8.97 | 6.28 [6.23, 6.36] |
| webqsp | 99.2 | 95.7 | 131.9 | **1.33 [1.27, 1.38]** | 1.38 [1.34, 1.44] | 0.86 [0.79, 0.90] | 1.54 [1.50, 1.62] | 1.54 [1.50, 1.63] | 84.69 -> 39.86 | 27.16 -> 17.70 | 3.63 [3.51, 3.75] |

Stage p50 ms:

- metaqa zrc6: pool 1.99, read 7.27, compile 9.22, edges 0.11, lean 7.89, chains 4.51, forward 8.97
- metaqa zsp6: pool 1.93, read 7.26, compile 9.22, edges 0.11, lean 7.91, chains 4.39, links 1.55, forward 9.64
- metaqa gnn5: pool 1.94, read 7.27, compile 31.71, pack 1.49, forward 54.18
- metaqa zrc5: pool 1.99, read 7.27, compile 9.20, edges 0.11, lean 7.89, chains 5.64, forward 9.71
- metaqa zsp5: pool 1.94, read 7.26, compile 9.20, edges 0.11, lean 7.93, chains 5.53, links 1.55, forward 10.42
- webqsp zrc6: pool 5.52, read 9.08, compile 16.66, edges 0.15, lean 9.22, chains 39.86, forward 17.70
- webqsp zsp6: pool 5.54, read 9.09, compile 16.57, edges 0.15, lean 9.24, chains 33.86, links 2.22, forward 18.43
- webqsp gnn5: pool 5.46, read 9.06, compile 48.71, pack 1.96, forward 66.37
- webqsp zrc5: pool 5.39, read 8.98, compile 16.62, edges 0.15, lean 9.20, chains 84.69, forward 27.16
- webqsp zsp5: pool 5.50, read 9.08, compile 16.69, edges 0.15, lean 9.23, chains 76.01, links 2.20, forward 27.90

Checks:

- metaqa: kept 200/200; {'pool': 200, 'seeds': 200, 'edges': 200, 'e_rel': 200, 'zrc5_entries': 200, 'zsp5_entries': 200, 'zrc5_within_tol': 200, 'zrc5_top5_same': 200, 'zsp5_within_tol': 200, 'zsp5_top5_same': 200, 'gnn5_top5_look': 200, 'q_emb': 200, 'questions': 200, 'zrc5_diff_max': 8.058547973632812e-05, 'zsp5_diff_max': 6.365776062011719e-05, 'gnn5_vs_look_max': 1.4781951904296875e-05, 'zrc5_vs_cache_max': 0.00189971923828125, 'zsp5_vs_cache_max': 0.000736236572265625, 'zrc5_vs_cache_top5': '200/200', 'zsp5_vs_cache_top5': '200/200', 'zrc5_rows_vs_cache': {'questions_differing': 19, 'elements': 19, 'largest': 0.000244140625}, 'zsp5_rows_vs_cache': {'questions_differing': 19, 'elements': 19, 'largest': 0.000244140625}, 'zrc5_n_entries_p50': 6408.0, 'zsp5_n_entries_p50': 6408.0, 'zrc6_entries_bits': 200, 'zsp6_entries_bits': 200, 'zrc6_within_tol': 200, 'zrc6_top5_same': 200, 'zsp6_within_tol': 200, 'zsp6_top5_same': 200, 'zrc6_diff_max': 8.058547973632812e-05, 'zsp6_diff_max': 6.365776062011719e-05}
- webqsp: kept 200/200; {'pool': 200, 'seeds': 200, 'edges': 200, 'e_rel': 200, 'zrc5_entries': 200, 'zsp5_entries': 200, 'zrc5_within_tol': 200, 'zrc5_top5_same': 200, 'zsp5_within_tol': 200, 'zsp5_top5_same': 200, 'gnn5_top5_look': 200, 'q_emb': 200, 'questions': 200, 'zrc5_diff_max': 8.7738037109375e-05, 'zsp5_diff_max': 7.724761962890625e-05, 'gnn5_vs_look_max': 8.106231689453125e-06, 'zrc5_vs_cache_max': 0.01441049575805664, 'zsp5_vs_cache_max': 0.011312246322631836, 'zrc5_vs_cache_top5': '200/200', 'zsp5_vs_cache_top5': '200/200', 'zrc5_rows_vs_cache': {'questions_differing': 10, 'elements': 24, 'largest': 0.000732421875}, 'zsp5_rows_vs_cache': {'questions_differing': 10, 'elements': 24, 'largest': 0.000732421875}, 'zrc5_n_entries_p50': 87298.5, 'zsp5_n_entries_p50': 87298.5, 'zrc6_entries_bits': 200, 'zsp6_entries_bits': 200, 'zrc6_within_tol': 200, 'zrc6_top5_same': 200, 'zsp6_within_tol': 200, 'zsp6_top5_same': 200, 'zrc6_diff_max': 8.7738037109375e-05, 'zsp6_diff_max': 7.724761962890625e-05}
