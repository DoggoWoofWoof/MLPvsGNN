P4a: KIND_T's pool cold, batch 1, one thread (p50 ms; ratios with 2,000-resample bootstrap 95% CIs). Call: **WALKS_DOMINATE** on metaqa, musique, webqsp, hotpotqa, 2wiki; next P4b.

| dataset | kept | KIND_T p50 | p95 | p99 | walks' share | B_q=0 | entity | route | shortcut | page faults | CPU load before | free GB before |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 200/200 | 0.19 | 0.35 | 0.45 | 0.00 | 200 | 0 | 0 | 0 | 0 | 8.8 | 5.6 |
| metaqa | 200/200 | 10.44 | 26.02 | 30.20 | 0.88 | 0 | 200 | 159 | 158 | 722 | 14.6 | 5.48 |
| musique | 200/200 | 75.98 | 110.94 | 116.45 | 0.92 | 0 | 0 | 87 | 0 | 1362 | 11.5 | 5.22 |
| webqsp | 200/200 | 61.32 | 102.60 | 145.58 | 0.96 | 0 | 188 | 55 | 54 | 24031 | 9.6 | 4.4 |
| hotpotqa | 200/200 | 286.24 | 378.71 | 404.12 | 0.97 | 0 | 0 | 2 | 0 | 40153 | 9.0 | 3.78 |
| 2wiki | 200/200 | 198.91 | 254.81 | 267.44 | 0.97 | 0 | 0 | 1 | 0 | 39569 | 18.1 | 2.82 |

End to end, cold p50 ms (C12's fastest forms; with KIND_T: C12's model stages on today's pool and KIND_T's build in place of today's pool stage):

| dataset | umlp today | umlp KIND_T | gnn today | gnn KIND_T | zrc today | zrc KIND_T | zsp today | zsp KIND_T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 2.67 | 2.81 | 4.24 | 4.38 | 1.65 | 1.77 | 1.73 | 1.87 |
| metaqa | 49.47 | 59.15 | 77.45 | 86.86 | 26.42 | 36.19 | 28.07 | 37.92 |
| musique | 75.66 | 150.79 | 143.26 | 218.77 | 35.87 | 111.02 | 42.19 | 116.42 |
| webqsp | 69.87 | 130.32 | 107.26 | 169.67 | 82.65 | 149.11 | 79.62 | 146.04 |
| hotpotqa | 3.91 | 290.01 | 5.88 | 291.62 | 2.36 | 288.57 | 2.48 | 288.62 |
| 2wiki | 4.11 | 203.59 | 6.00 | 205.12 | 2.52 | 201.56 | 2.61 | 201.67 |

Ratios on today's pool and with KIND_T's build:

| dataset | pool | universal | lean | fastest_gnn_over_universal_mlp | fastest_gnn_over_zrc | universal_gnn_over_zrc |
| --- | --- | --- | --- | --- | --- | --- |
| squad | today | 1.59 [1.55, 1.69] | 1.05 [1.03, 1.09] | 0.65 [0.64, 0.68] | 1.05 [1.03, 1.09] | 2.57 [2.50, 2.72] |
| squad | kind_t | 1.56 [1.52, 1.65] | 1.05 [1.03, 1.09] | 0.67 [0.65, 0.69] | 1.05 [1.03, 1.09] | 2.47 [2.38, 2.61] |
| metaqa | today | 1.57 [1.55, 1.57] | 1.06 [1.06, 1.07] | 0.57 [0.56, 0.57] | 1.06 [1.06, 1.07] | 2.93 [2.91, 2.95] |
| metaqa | kind_t | 1.47 [1.45, 1.48] | 1.05 [1.04, 1.06] | 0.64 [0.63, 0.65] | 1.05 [1.04, 1.06] | 2.40 [2.36, 2.44] |
| musique | today | 1.89 [1.87, 1.91] | 1.18 [1.16, 1.19] | 0.56 [0.55, 0.56] | 1.18 [1.16, 1.19] | 3.99 [3.94, 4.03] |
| musique | kind_t | 1.45 [1.43, 1.47] | 1.05 [1.05, 1.06] | 0.77 [0.77, 0.78] | 1.05 [1.05, 1.06] | 1.97 [1.93, 2.00] |
| webqsp | today | 1.54 [1.51, 1.55] | 0.96 [0.93, 0.99] | 1.14 [1.09, 1.19] | 0.96 [0.93, 0.99] | 1.30 [1.24, 1.34] |
| webqsp | kind_t | 1.30 [1.27, 1.32] | 0.98 [0.95, 0.99] | 1.12 [1.06, 1.15] | 0.98 [0.95, 0.99] | 1.14 [1.11, 1.18] |
| hotpotqa | today | 1.50 [1.44, 1.56] | 1.05 [1.01, 1.09] | 0.63 [0.61, 0.66] | 1.05 [1.01, 1.09] | 2.49 [2.38, 2.57] |
| hotpotqa | kind_t | 1.01 [1.00, 1.01] | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.01 [1.01, 1.01] |
| 2wiki | today | 1.46 [1.43, 1.50] | 1.04 [1.01, 1.07] | 0.63 [0.62, 0.65] | 1.04 [1.01, 1.07] | 2.38 [2.33, 2.45] |
| 2wiki | kind_t | 1.01 [1.01, 1.01] | 1.00 [1.00, 1.00] | 0.99 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.02 [1.01, 1.02] |

KIND_T's build as a share of each model's end-to-end (median over questions):

- squad: umlp 0.07, gnn 0.04, zrc 0.11, zsp 0.10; build p50 / zrc's end-to-end p50 0.11
- metaqa: umlp 0.19, gnn 0.13, zrc 0.30, zsp 0.29; build p50 / zrc's end-to-end p50 0.29
- musique: umlp 0.50, gnn 0.35, zrc 0.68, zsp 0.65; build p50 / zrc's end-to-end p50 0.68
- webqsp: umlp 0.48, gnn 0.36, zrc 0.41, zsp 0.43; build p50 / zrc's end-to-end p50 0.41
- hotpotqa: umlp 0.99, gnn 0.98, zrc 0.99, zsp 0.99; build p50 / zrc's end-to-end p50 0.99
- 2wiki: umlp 0.98, gnn 0.97, zrc 0.99, zsp 0.99; build p50 / zrc's end-to-end p50 0.99

Stages (questions run on, p50 ms where run, share of the build):

- squad: bs 200, 0.086, 0.47; pool 200, 0.103, 0.53
  walks run {'Kst': 0, 'A1st': 0, 'A1all': 0, 'KTst': 0, 'KTall': 0}; touched p50 {}; today's pool stage p50 p4a 0.103, c12_umlp 0.073, c12_gnn 0.075, c12_zrc 0.077, c12_zsp 0.073
- metaqa: bs 200, 0.099, 0.01; pool 200, 0.297, 0.03; fill 200, 0.631, 0.06; kind 200, 0.182, 0.02; link 200, 0.052, 0.00; order 42, 0.005, 0.00; restarts 200, 0.016, 0.01; w_a1st 42, 21.730, 0.36; w_kst 159, 8.157, 0.51
  walks run {'Kst': 159, 'A1st': 42, 'A1all': 0, 'KTst': 0, 'KTall': 0}; touched p50 {'Kst': 31895.0, 'A1st': 43120.5}; today's pool stage p50 p4a 0.297, c12_umlp 0.246, c12_gnn 0.251, c12_zrc 0.254, c12_zsp 0.243
- musique: bs 200, 0.117, 0.00; pool 200, 0.419, 0.01; fill 200, 0.893, 0.01; kind 200, 0.177, 0.00; link 200, 0.093, 0.00; order 200, 3.787, 0.05; restarts 200, 0.732, 0.01; w_a1all 200, 28.019, 0.37; w_a1st 200, 24.818, 0.33; w_ktall 154, 11.436, 0.12; w_ktst 154, 9.235, 0.10
  walks run {'Kst': 0, 'A1st': 200, 'A1all': 200, 'KTst': 154, 'KTall': 154}; touched p50 {'A1st': 82093.5, 'A1all': 90988.5, 'KTst': 36152.5, 'KTall': 52594.0}; today's pool stage p50 p4a 0.419, c12_umlp 0.324, c12_gnn 0.325, c12_zrc 0.328, c12_zsp 0.321
- webqsp: bs 200, 0.059, 0.00; pool 200, 0.479, 0.01; fill 200, 1.132, 0.02; kind 200, 0.206, 0.00; link 200, 0.087, 0.00; order 146, 0.008, 0.00; restarts 200, 0.354, 0.00; w_a1all 12, 52.341, 0.05; w_a1st 146, 64.664, 0.75; w_kst 55, 36.922, 0.16
  walks run {'Kst': 55, 'A1st': 146, 'A1all': 12, 'KTst': 0, 'KTall': 0}; touched p50 {'Kst': 164291.0, 'A1st': 219648.5, 'A1all': 154969.0}; today's pool stage p50 p4a 0.479, c12_umlp 0.285, c12_gnn 0.290, c12_zrc 0.292, c12_zsp 0.284
- hotpotqa: bs 200, 0.089, 0.00; pool 200, 0.144, 0.00; fill 200, 0.662, 0.00; kind 200, 0.187, 0.00; link 200, 0.190, 0.00; order 200, 4.910, 0.02; restarts 200, 0.789, 0.00; w_a1all 200, 104.961, 0.37; w_a1st 200, 113.638, 0.39; w_ktall 178, 34.323, 0.11; w_ktst 178, 29.701, 0.11
  walks run {'Kst': 0, 'A1st': 200, 'A1all': 200, 'KTst': 178, 'KTall': 178}; touched p50 {'A1st': 267240.0, 'A1all': 272717.5, 'KTst': 55121.0, 'KTall': 79411.0}; today's pool stage p50 p4a 0.144, c12_umlp 0.088, c12_gnn 0.092, c12_zrc 0.092, c12_zsp 0.087
- 2wiki: bs 200, 0.067, 0.00; pool 200, 0.105, 0.00; fill 200, 0.504, 0.00; kind 200, 0.133, 0.00; link 200, 0.078, 0.00; order 200, 3.766, 0.02; restarts 200, 0.607, 0.00; w_a1all 200, 68.541, 0.35; w_a1st 200, 72.406, 0.36; w_ktall 200, 25.326, 0.13; w_ktst 200, 25.575, 0.14
  walks run {'Kst': 0, 'A1st': 200, 'A1all': 200, 'KTst': 200, 'KTall': 200}; touched p50 {'A1st': 321239.5, 'A1all': 326804.0, 'KTst': 94842.5, 'KTall': 112435.5}; today's pool stage p50 p4a 0.105, c12_umlp 0.088, c12_gnn 0.090, c12_zrc 0.092, c12_zsp 0.086

Checks:

- squad: kept 200/200; every check held on every question; paired with C12's kept questions: 200
- metaqa: kept 200/200; every check held on every question; paired with C12's kept questions: 200
- musique: kept 200/200; every check held on every question; paired with C12's kept questions: 200
- webqsp: kept 200/200; every check held on every question; paired with C12's kept questions: 200
- hotpotqa: kept 200/200; every check held on every question; paired with C12's kept questions: 200
- 2wiki: kept 200/200; every check held on every question; paired with C12's kept questions: 200
