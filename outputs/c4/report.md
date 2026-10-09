Cold, batch 1, total per question, p50 ms:

| dataset | zrc4 | zsp4 | gnn4 | **gnn4 / zrc4** | gnn4 / zsp4 | gnn3 / zrc3 (C3, same run) | zrc3 / zrc4 | gnn3 / gnn4 | compile p50 zrc3 -> zrc4 | gnn3 -> gnn4 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 4.5 | 4.8 | 9.2 | **2.05 [2.00, 2.11]** | 1.90 [1.86, 1.95] | 1.85 [1.80, 1.93] | 1.38 [1.31, 1.44] | 1.25 [1.21, 1.29] | 2.59 -> 1.14 | 4.53 -> 2.71 |
| hotpotqa | 3.9 | 4.2 | 7.7 | **2.00 [1.96, 2.06]** | 1.86 [1.82, 1.91] | 1.92 [1.83, 2.02] | 1.34 [1.27, 1.46] | 1.29 [1.24, 1.35] | 2.03 -> 0.92 | 3.81 -> 2.22 |
| musique | 51.5 | 58.2 | 176.7 | **3.43 [3.39, 3.47]** | 3.03 [2.99, 3.06] | 3.40 [3.37, 3.44] | 1.00 [0.99, 1.01] | 0.99 [0.99, 1.01] | 16.66 -> 16.71 | 58.99 -> 59.02 |
| squad | 2.5 | 2.8 | 5.6 | **2.27 [2.22, 2.34]** | 2.01 [1.95, 2.06] | 2.29 [2.25, 2.36] | 0.98 [0.97, 1.00] | 0.99 [0.97, 1.02] | 0.58 -> 0.61 | 1.54 -> 1.57 |

Checks:

- 2wiki: kept 200/200; {'questions': 200, 'zrc4_compile_bits_ok': 200, 'zrc4_scores_equal_ok': 200, 'zsp4_compile_bits_ok': 200, 'zsp4_scores_equal_ok': 200, 'gnn4_compile_bits_ok': 200, 'gnn4_scores_equal_ok': 200, 'gnn4_top5_look_ok': 200, 'pool_ok': 200, 'gnn4_vs_look_max': 3.814697265625e-06}
- hotpotqa: kept 200/200; {'questions': 200, 'zrc4_compile_bits_ok': 200, 'zrc4_scores_equal_ok': 200, 'zsp4_compile_bits_ok': 200, 'zsp4_scores_equal_ok': 200, 'gnn4_compile_bits_ok': 200, 'gnn4_scores_equal_ok': 200, 'gnn4_top5_look_ok': 200, 'pool_ok': 200, 'gnn4_vs_look_max': 3.814697265625e-06}
- musique: kept 200/200; {'questions': 200, 'zrc4_compile_bits_ok': 200, 'zrc4_scores_equal_ok': 200, 'zsp4_compile_bits_ok': 200, 'zsp4_scores_equal_ok': 200, 'gnn4_compile_bits_ok': 200, 'gnn4_scores_equal_ok': 200, 'gnn4_top5_look_ok': 200, 'pool_ok': 200, 'gnn4_vs_look_max': 4.76837158203125e-06}
- squad: kept 200/200; {'questions': 200, 'zrc4_compile_bits_ok': 200, 'zrc4_scores_equal_ok': 200, 'zsp4_compile_bits_ok': 200, 'zsp4_scores_equal_ok': 200, 'gnn4_compile_bits_ok': 200, 'gnn4_scores_equal_ok': 200, 'gnn4_top5_look_ok': 200, 'pool_ok': 200, 'gnn4_vs_look_max': 2.86102294921875e-06}
