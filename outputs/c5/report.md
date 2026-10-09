Cold, batch 1, total per question, p50 ms:

| dataset | pool rows p50 | zrc5 | zsp5 | gnn5 | **gnn5 / zrc5** | gnn5 / zsp5 | zrc5 chains p50 | warm forward gnn5 / zrc5 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| metaqa | 2038 | 42.0 | 44.1 | 98.1 | **2.34 [2.32, 2.36]** | 2.22 [2.20, 2.25] | 5.62 | 5.91 [5.83, 6.00] |
| webqsp | 2122 | 152.6 | 145.3 | 127.8 | **0.84 [0.78, 0.89]** | 0.88 [0.81, 0.93] | 83.12 | 2.34 [2.29, 2.40] |

Stage p50 ms:

- metaqa zrc5: pool 1.98, read 7.24, compile 9.04, edges 0.11, lean 7.79, chains 5.62, forward 9.59
- metaqa zsp5: pool 1.92, read 7.24, compile 9.09, edges 0.11, lean 7.82, chains 5.41, links 1.53, forward 10.30
- metaqa gnn5: pool 1.94, read 7.22, compile 31.34, pack 1.48, forward 55.77
- webqsp zrc5: pool 5.35, read 8.98, compile 16.32, edges 0.15, lean 9.11, chains 83.12, forward 26.39
- webqsp zsp5: pool 5.41, read 9.06, compile 16.35, edges 0.15, lean 9.19, chains 73.40, links 2.17, forward 27.44
- webqsp gnn5: pool 5.42, read 8.98, compile 47.52, pack 1.93, forward 63.38

Checks:

- metaqa: kept 200/200; {'pool': 200, 'seeds': 200, 'edges': 200, 'e_rel': 200, 'zrc5_entries': 200, 'zsp5_entries': 200, 'zrc5_within_tol': 200, 'zrc5_top5_same': 200, 'zsp5_within_tol': 200, 'zsp5_top5_same': 200, 'gnn5_top5_look': 200, 'q_emb': 200, 'questions': 200, 'zrc5_diff_max': 8.058547973632812e-05, 'zsp5_diff_max': 6.365776062011719e-05, 'gnn5_vs_look_max': 1.4781951904296875e-05, 'zrc5_vs_cache_max': 0.00189971923828125, 'zsp5_vs_cache_max': 0.000736236572265625, 'zrc5_vs_cache_top5': '200/200', 'zsp5_vs_cache_top5': '200/200', 'zrc5_rows_vs_cache': {'questions_differing': 19, 'elements': 19, 'largest': 0.000244140625}, 'zsp5_rows_vs_cache': {'questions_differing': 19, 'elements': 19, 'largest': 0.000244140625}, 'zrc5_n_entries_p50': 6408.0, 'zsp5_n_entries_p50': 6408.0}
- webqsp: kept 200/200; {'pool': 200, 'seeds': 200, 'edges': 200, 'e_rel': 200, 'zrc5_entries': 200, 'zsp5_entries': 200, 'zrc5_within_tol': 200, 'zrc5_top5_same': 200, 'zsp5_within_tol': 200, 'zsp5_top5_same': 200, 'gnn5_top5_look': 200, 'q_emb': 200, 'questions': 200, 'zrc5_diff_max': 8.7738037109375e-05, 'zsp5_diff_max': 7.724761962890625e-05, 'gnn5_vs_look_max': 8.106231689453125e-06, 'zrc5_vs_cache_max': 0.01441049575805664, 'zsp5_vs_cache_max': 0.011312246322631836, 'zrc5_vs_cache_top5': '200/200', 'zsp5_vs_cache_top5': '200/200', 'zrc5_rows_vs_cache': {'questions_differing': 10, 'elements': 24, 'largest': 0.000732421875}, 'zsp5_rows_vs_cache': {'questions_differing': 10, 'elements': 24, 'largest': 0.000732421875}, 'zrc5_n_entries_p50': 87298.5, 'zsp5_n_entries_p50': 87298.5}
