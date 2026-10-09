Cold, batch 1, total per question, p50 ms:

| dataset | zrc3 (MLP) | zsp3 (GNN track) | gnn3 (six GNN) | gnn3 / zrc3 | gnn3 / zsp3 | zrc2 | gnn6 | zrc2 / zrc3 | gnn6 / gnn3 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| 2wiki | 6.2 | 7.0 | 11.4 | 1.84 [1.73, 1.90] | 1.62 [1.58, 1.70] | 9.7 | 12.5 | 1.57 [1.48, 1.62] | 1.10 [1.05, 1.14] |
| hotpotqa | 5.7 | 6.2 | 10.4 | 1.84 [1.72, 1.95] | 1.67 [1.59, 1.74] | 8.9 | 11.3 | 1.57 [1.50, 1.66] | 1.09 [1.05, 1.14] |
| musique | 52.9 | 60.1 | 179.3 | 3.39 [3.34, 3.41] | 2.98 [2.95, 3.01] | 65.7 | 330.6 | 1.24 [1.23, 1.25] | 1.84 [1.83, 1.87] |
| squad | 2.6 | 3.0 | 5.9 | 2.23 [2.17, 2.31] | 1.98 [1.94, 2.05] | 5.5 | 6.9 | 2.10 [2.04, 2.15] | 1.17 [1.14, 1.25] |

Warm forward per question, p50 ms (inputs built; batch 1 and batch 16):

| dataset | zrc3 b1 | zsp3 b1 | gnn3 b1 | gnn3 / zrc3 b1 | zrc3 b16 | zsp3 b16 | gnn3 b16 | gnn3 / zrc3 b16 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2wiki | 0.91 | 1.23 | 4.34 | 4.77 [4.62, 4.90] | 0.58 | 0.62 | 3.07 | 5.3x |
| hotpotqa | 0.84 | 1.16 | 4.04 | 4.84 [4.68, 5.06] | 0.51 | 0.55 | 2.97 | 5.8x |
| musique | 8.11 | 9.36 | 102.91 | 12.69 [12.50, 12.88] | 11.95 | 14.39 | 111.32 | 9.3x |
| squad | 0.60 | 0.88 | 3.16 | 5.28 [5.09, 5.56] | 0.31 | 0.33 | 1.84 | 6.0x |

Checks (questions passing / measured):

- 2wiki: kept 200/200; {'questions': 200, 'pool_ok': 200, 'seeds_ok': 200, 'rows_bits_ok': 200, 'links_bits_ok': 200, 'zrc3_within_tol_ok': 200, 'zrc3_top5_same_ok': 200, 'zsp3_within_tol_ok': 200, 'zsp3_top5_same_ok': 200, 'gnn3_within_tol_ok': 200, 'gnn3_top5_same_ok': 200, 'gnn6_top5_same_ok': 200, 'zrc3_diff_max': 7.62939453125e-06, 'zsp3_diff_max': 5.245208740234375e-06, 'gnn3_diff_max': 1.9073486328125e-06, 'gnn6_vs_look_max': 3.814697265625e-06}
- hotpotqa: kept 200/200; {'questions': 200, 'pool_ok': 200, 'seeds_ok': 200, 'rows_bits_ok': 200, 'links_bits_ok': 200, 'zrc3_within_tol_ok': 200, 'zrc3_top5_same_ok': 200, 'zsp3_within_tol_ok': 200, 'zsp3_top5_same_ok': 200, 'gnn3_within_tol_ok': 200, 'gnn3_top5_same_ok': 200, 'gnn6_top5_same_ok': 200, 'zrc3_diff_max': 7.62939453125e-06, 'zsp3_diff_max': 5.7220458984375e-06, 'gnn3_diff_max': 2.86102294921875e-06, 'gnn6_vs_look_max': 3.814697265625e-06}
- musique: kept 200/200; {'questions': 200, 'pool_ok': 200, 'seeds_ok': 200, 'rows_bits_ok': 200, 'links_bits_ok': 200, 'zrc3_within_tol_ok': 200, 'zrc3_top5_same_ok': 200, 'zsp3_within_tol_ok': 200, 'zsp3_top5_same_ok': 200, 'gnn3_within_tol_ok': 200, 'gnn3_top5_same_ok': 200, 'gnn6_top5_same_ok': 200, 'zrc3_diff_max': 8.511543273925781e-05, 'zsp3_diff_max': 5.555152893066406e-05, 'gnn3_diff_max': 3.814697265625e-06, 'gnn6_vs_look_max': 5.245208740234375e-06}
- squad: kept 200/200; {'questions': 200, 'pool_ok': 200, 'seeds_ok': 200, 'rows_bits_ok': 200, 'links_bits_ok': 200, 'zrc3_within_tol_ok': 200, 'zrc3_top5_same_ok': 200, 'zsp3_within_tol_ok': 200, 'zsp3_top5_same_ok': 200, 'gnn3_within_tol_ok': 200, 'gnn3_top5_same_ok': 200, 'gnn6_top5_same_ok': 200, 'zrc3_diff_max': 3.5762786865234375e-06, 'zsp3_diff_max': 2.562999725341797e-06, 'gnn3_diff_max': 1.9073486328125e-06, 'gnn6_vs_look_max': 3.337860107421875e-06}
