P4b: KIND_T's build with P4a's walks and with P4b's form, both on every question, cold, batch 1, one thread (p50 ms; ratios P4a's form / P4b's form with 2,000-resample paired bootstrap 95% CIs). Call: **FASTER** (metaqa, musique, webqsp, hotpotqa, 2wiki).

The fresh set (questions 201-400, deciding):

| dataset | kept | walked | build P4a | build P4b | build ratio | walks P4a | walks P4b | walks ratio | CPU load before | free GB before | page faults |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| squad | 200/200 | 0 | 0.12 | 0.12 | 0.97 [0.85, 1.07] | - | - | - | 5.5 | 5.86 | 0 |
| metaqa | 200/200 | 200 | 7.24 | 6.36 | 1.14 [1.13, 1.18] | 6.36 | 5.40 | 1.18 [1.15, 1.20] | 5.6 | 5.66 | 1736 |
| musique | 200/200 | 200 | 48.93 | 34.40 | 1.42 [1.40, 1.46] | 44.46 | 29.84 | 1.49 [1.46, 1.53] | 7.8 | 5.58 | 1327 |
| webqsp | 200/200 | 200 | 34.58 | 13.20 | 2.62 [2.55, 2.74] | 33.00 | 11.63 | 2.84 [2.74, 2.96] | 8.8 | 4.28 | 22591 |
| hotpotqa | 200/200 | 200 | 171.05 | 47.72 | 3.58 [3.47, 3.64] | 165.63 | 42.91 | 3.86 [3.74, 3.93] | 8.5 | 2.6 | 89196 |
| 2wiki | 200/200 | 200 | 188.47 | 50.30 | 3.75 [3.68, 3.81] | 183.16 | 45.46 | 4.03 [3.95, 4.11] | 4.1 | 1.33 | 36877 |

The c12 set (questions 1-200, the tuning set):

| dataset | kept | walked | build P4a | build P4b | build ratio | walks P4a | walks P4b | walks ratio | CPU load before | free GB before | page faults |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| squad | 200/200 | 0 | 0.12 | 0.11 | 1.01 [0.87, 1.16] | - | - | - | 4.5 | 5.86 | 0 |
| metaqa | 200/200 | 200 | 7.37 | 6.29 | 1.17 [1.14, 1.21] | 6.49 | 5.45 | 1.19 [1.17, 1.24] | 10.7 | 5.66 | 3364 |
| musique | 200/200 | 200 | 51.13 | 35.13 | 1.46 [1.42, 1.48] | 46.55 | 30.65 | 1.52 [1.48, 1.55] | 6.8 | 5.57 | 3287 |
| webqsp | 200/200 | 200 | 37.94 | 13.21 | 2.87 [2.75, 2.95] | 36.21 | 11.76 | 3.08 [2.97, 3.19] | 3.8 | 4.27 | 26508 |
| hotpotqa | 200/200 | 200 | 167.13 | 47.28 | 3.54 [3.48, 3.64] | 161.62 | 42.58 | 3.80 [3.73, 3.95] | 7.2 | 2.84 | 41393 |
| 2wiki | 200/200 | 200 | 188.11 | 50.80 | 3.70 [3.63, 3.79] | 182.96 | 46.03 | 3.98 [3.89, 4.07] | 4.2 | 1.35 | 38648 |

End to end on C12's set, cold p50 ms (C12's model stages on today's pool, KIND_T's build in place of today's pool stage; P4a's caveat):

| dataset | umlp today | umlp P4a | umlp P4b | gnn today | gnn P4a | gnn P4b | zrc today | zrc P4a | zrc P4b | zsp today | zsp P4a | zsp P4b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 2.67 | 2.71 | 2.71 | 4.24 | 4.28 | 4.30 | 1.65 | 1.67 | 1.69 | 1.73 | 1.77 | 1.79 |
| metaqa | 49.47 | 56.34 | 55.44 | 77.45 | 84.16 | 83.13 | 26.42 | 33.20 | 32.26 | 28.07 | 34.86 | 34.01 |
| musique | 75.66 | 125.18 | 109.29 | 143.26 | 192.40 | 176.81 | 35.87 | 85.52 | 70.33 | 42.19 | 91.97 | 76.39 |
| webqsp | 69.87 | 105.93 | 83.24 | 107.26 | 145.38 | 120.46 | 82.65 | 121.74 | 96.97 | 79.62 | 118.73 | 93.41 |
| hotpotqa | 3.91 | 170.92 | 51.12 | 5.88 | 173.13 | 53.40 | 2.36 | 169.19 | 49.38 | 2.48 | 169.12 | 49.37 |
| 2wiki | 4.11 | 191.99 | 54.71 | 6.00 | 194.50 | 56.60 | 2.52 | 190.48 | 53.32 | 2.61 | 190.41 | 53.55 |

Ratios on C12's set with each form's build:

| dataset | form | universal | lean | fastest_gnn_over_universal_mlp | fastest_gnn_over_zrc | universal_gnn_over_zrc |
| --- | --- | --- | --- | --- | --- | --- |
| squad | p4a | 1.58 [1.54, 1.67] | 1.06 [1.04, 1.10] | 0.65 [0.64, 0.68] | 1.06 [1.04, 1.10] | 2.56 [2.49, 2.71] |
| squad | p4b | 1.59 [1.54, 1.68] | 1.06 [1.03, 1.10] | 0.66 [0.64, 0.68] | 1.06 [1.04, 1.10] | 2.55 [2.47, 2.70] |
| metaqa | p4a | 1.49 [1.48, 1.51] | 1.05 [1.04, 1.06] | 0.62 [0.61, 0.63] | 1.05 [1.04, 1.06] | 2.54 [2.50, 2.57] |
| metaqa | p4b | 1.50 [1.49, 1.51] | 1.05 [1.04, 1.06] | 0.61 [0.61, 0.62] | 1.05 [1.04, 1.06] | 2.58 [2.55, 2.60] |
| musique | p4a | 1.54 [1.50, 1.56] | 1.08 [1.06, 1.09] | 0.73 [0.73, 0.74] | 1.08 [1.06, 1.08] | 2.25 [2.19, 2.29] |
| musique | p4b | 1.62 [1.59, 1.63] | 1.09 [1.08, 1.10] | 0.70 [0.68, 0.71] | 1.09 [1.08, 1.10] | 2.51 [2.47, 2.59] |
| webqsp | p4a | 1.37 [1.34, 1.39] | 0.98 [0.95, 0.99] | 1.12 [1.07, 1.16] | 0.98 [0.95, 0.99] | 1.19 [1.14, 1.23] |
| webqsp | p4b | 1.45 [1.43, 1.47] | 0.96 [0.93, 0.98] | 1.12 [1.07, 1.16] | 0.96 [0.93, 0.98] | 1.24 [1.20, 1.28] |
| hotpotqa | p4a | 1.01 [1.01, 1.01] | 1.00 [1.00, 1.00] | 0.99 [0.99, 0.99] | 1.00 [1.00, 1.00] | 1.02 [1.02, 1.03] |
| hotpotqa | p4b | 1.04 [1.03, 1.05] | 1.00 [1.00, 1.02] | 0.97 [0.96, 0.98] | 1.00 [1.00, 1.02] | 1.08 [1.06, 1.08] |
| 2wiki | p4a | 1.01 [1.00, 1.02] | 1.00 [1.00, 1.00] | 0.99 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.02 [1.01, 1.02] |
| 2wiki | p4b | 1.03 [1.03, 1.04] | 1.00 [0.99, 1.01] | 0.98 [0.97, 0.99] | 1.00 [1.00, 1.01] | 1.06 [1.05, 1.07] |

Each walk's p50 where it ran (ms, P4a's form -> P4b's form), the build's p50 by position, and the form's index time and memory:

- squad, fresh: no walk; by position p4a first 0.13, second 0.09; p4b first 0.14, second 0.09
- squad, c12: no walk; by position p4a first 0.13, second 0.09; p4b first 0.13, second 0.09
  index: numbering 0.6s, form 0.2s, 0.02 GB (P4a's buffers 0.00 GB); P4a's filed build p50 0.19
- metaqa, fresh: Kst 194x 6.21 -> 5.37; A1st 7x 15.03 -> 13.67; by position p4a first 7.23, second 7.34; p4b first 6.34, second 6.40
- metaqa, c12: Kst 159x 5.93 -> 4.98; A1st 42x 15.18 -> 13.98; by position p4a first 7.37, second 7.37; p4b first 6.28, second 6.29
  index: numbering 0.6s, form 0.2s, 0.01 GB (P4a's buffers 0.00 GB); P4a's filed build p50 10.44
- musique, fresh: A1st 200x 16.66 -> 11.09; A1all 200x 18.47 -> 10.90; KTst 138x 6.69 -> 5.38; KTall 138x 7.36 -> 5.60; by position p4a first 49.06, second 48.65; p4b first 34.03, second 34.53
- musique, c12: A1st 200x 17.04 -> 11.06; A1all 200x 18.82 -> 10.82; KTst 154x 6.40 -> 5.03; KTall 154x 7.58 -> 5.80; by position p4a first 51.82, second 49.95; p4b first 34.81, second 35.33
  index: numbering 0.6s, form 0.2s, 0.06 GB (P4a's buffers 0.00 GB); P4a's filed build p50 75.98
- webqsp, fresh: Kst 61x 19.13 -> 7.42; A1st 139x 37.17 -> 12.66; A1all 5x 28.73 -> 8.87; by position p4a first 34.21, second 35.25; p4b first 13.52, second 12.91
- webqsp, c12: Kst 55x 22.34 -> 7.52; A1st 146x 38.33 -> 12.76; A1all 12x 30.26 -> 10.45; by position p4a first 38.34, second 37.25; p4b first 13.38, second 13.09
  index: numbering 1.1s, form 1.4s, 0.42 GB (P4a's buffers 0.09 GB); P4a's filed build p50 61.32
- hotpotqa, fresh: A1st 200x 64.91 -> 15.32; A1all 200x 60.84 -> 14.92; KTst 181x 18.66 -> 6.33; KTall 181x 21.43 -> 6.58; by position p4a first 173.82, second 170.18; p4b first 47.62, second 48.42
- hotpotqa, c12: A1st 200x 65.22 -> 15.62; A1all 200x 61.74 -> 14.99; KTst 178x 18.18 -> 5.95; KTall 178x 20.57 -> 6.49; by position p4a first 167.44, second 166.83; p4b first 46.85, second 47.75
  index: numbering 2.4s, form 4.2s, 1.00 GB (P4a's buffers 0.17 GB); P4a's filed build p50 286.24
- 2wiki, fresh: A1st 200x 66.86 -> 15.09; A1all 200x 66.04 -> 15.88; KTst 200x 23.83 -> 7.04; KTall 200x 25.47 -> 7.06; by position p4a first 191.19, second 187.03; p4b first 49.66, second 50.97
- 2wiki, c12: A1st 200x 67.41 -> 15.30; A1all 200x 64.31 -> 15.78; KTst 200x 24.27 -> 7.39; KTall 200x 24.43 -> 6.93; by position p4a first 191.38, second 185.43; p4b first 50.18, second 50.91
  index: numbering 3.0s, form 6.4s, 1.41 GB (P4a's buffers 0.20 GB); P4a's filed build p50 198.91

Checks:

- squad, fresh: kept 200/200; walks compared 0; every check held on every question
- squad, c12: kept 200/200; walks compared 0; every check held on every question
- metaqa, fresh: kept 200/200; walks compared 201; every check held on every question
- metaqa, c12: kept 200/200; walks compared 201; every check held on every question
- musique, fresh: kept 200/200; walks compared 676; every check held on every question
- musique, c12: kept 200/200; walks compared 708; every check held on every question
- webqsp, fresh: kept 200/200; walks compared 205; every check held on every question
- webqsp, c12: kept 200/200; walks compared 213; every check held on every question
- hotpotqa, fresh: kept 200/200; walks compared 762; every check held on every question
- hotpotqa, c12: kept 200/200; walks compared 756; every check held on every question
- 2wiki, fresh: kept 200/200; walks compared 800; every check held on every question
- 2wiki, c12: kept 200/200; walks compared 800; every check held on every question
