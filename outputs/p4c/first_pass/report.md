P4c: the KT input's serving path beside KIND_T's build, both on every question, cold, batch 1, one thread, P4b's walks (p50 ms; ratios serve / build with 2,000-resample paired bootstrap 95% CIs). Call: **TIMED**.

The fresh set (questions 201-400):

| dataset | kept | build | serve | head | serve / build | serve + head / build | extra p50 | rows p50 | B_q<=0 | entity | CPU load before | free GB before | page faults |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 200/200 | 0.15 | 22.20 | 0.57 | 145.83 [142.14, 166.73] | 149.76 [145.63, 171.81] | 22.07 | 50 | 200 | 0 | 27.9 | 5.61 | 128 |
| metaqa | 200/200 | 6.61 | 45.95 | 0.70 | 6.96 [6.73, 7.14] | 7.06 [6.83, 7.27] | 39.47 | 2036 | 0 | 200 | 10.6 | 5.75 | 9082 |
| musique | 200/200 | 35.10 | 40.16 | 0.75 | 1.14 [1.08, 1.16] | 1.16 [1.11, 1.18] | 3.76 | 2136 | 0 | 0 | 23.4 | 5.54 | 13010 |
| webqsp | 200/200 | 12.97 | 43.06 | 0.73 | 3.32 [3.13, 3.51] | 3.38 [3.19, 3.58] | 31.40 | 2126 | 0 | 195 | 7.0 | 4.42 | 12374 |
| hotpotqa | 200/200 | 48.97 | 56.21 | 0.56 | 1.15 [1.12, 1.16] | 1.16 [1.14, 1.17] | 7.20 | 90 | 0 | 0 | 6.7 | 2.92 | 9220 |
| 2wiki | 200/200 | 52.35 | 59.49 | 0.58 | 1.14 [1.10, 1.15] | 1.15 [1.12, 1.16] | 7.14 | 105 | 0 | 0 | 9.0 | 1.96 | 6171 |

The c12 set (questions 1-200):

| dataset | kept | build | serve | head | serve / build | serve + head / build | extra p50 | rows p50 | B_q<=0 | entity | CPU load before | free GB before | page faults |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 200/200 | 0.15 | 25.80 | 0.54 | 171.94 [167.31, 206.87] | 175.46 [170.80, 212.50] | 25.68 | 50 | 200 | 0 | 20.2 | 5.64 | 86 |
| metaqa | 200/200 | 6.41 | 44.23 | 0.70 | 6.90 [6.53, 7.24] | 7.02 [6.58, 7.39] | 37.52 | 2038 | 0 | 200 | 19.7 | 5.66 | 9976 |
| musique | 200/200 | 36.63 | 40.70 | 0.70 | 1.11 [1.09, 1.14] | 1.13 [1.11, 1.16] | 4.17 | 2142 | 0 | 0 | 18.3 | 5.49 | 8797 |
| webqsp | 200/200 | 12.98 | 40.81 | 0.66 | 3.14 [2.98, 3.34] | 3.20 [3.03, 3.40] | 29.39 | 2122 | 0 | 188 | 14.4 | 4.32 | 7656 |
| hotpotqa | 200/200 | 47.12 | 53.77 | 0.54 | 1.14 [1.13, 1.18] | 1.15 [1.14, 1.19] | 6.74 | 89 | 0 | 0 | 11.5 | 2.89 | 908 |
| 2wiki | 200/200 | 52.19 | 60.88 | 0.55 | 1.17 [1.14, 1.18] | 1.18 [1.15, 1.20] | 7.61 | 103 | 0 | 0 | 4.7 | 1.93 | 1513 |

End to end on C12's set, cold p50 ms: C12's model stages (C12's session, today's pools) plus the build timed here (umlp = twin0, gnn = gnn0, zrc, zsp) or plus the serve path and the head (zrck on zrc's stages, zspk on zsp's):

| dataset | umlp | gnn | zrc | zsp | zrck | zspk | fastest GNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| squad | 2.73 | 4.34 | 1.72 | 1.82 | 28.06 | 28.17 | gnn |
| metaqa | 55.41 | 83.29 | 32.32 | 34.02 | 69.74 | 71.55 | zspk |
| musique | 110.02 | 177.65 | 71.04 | 77.16 | 76.48 | 82.19 | zspk |
| webqsp | 82.95 | 120.47 | 96.31 | 92.75 | 129.00 | 125.83 | gnn |
| hotpotqa | 51.03 | 53.05 | 49.24 | 49.55 | 57.11 | 57.26 | gnn |
| 2wiki | 56.31 | 58.00 | 54.67 | 54.82 | 63.86 | 63.86 | gnn |

Ratios on C12's set (paired bootstrap 95% CIs):

| dataset | universal | lean_on_build | lean_kt | fastest_gnn_over_zrck | universal_gnn_over_zrck | fastest_gnn_over_universal_mlp |
| --- | --- | --- | --- | --- | --- | --- |
| squad | 1.59 [1.53, 1.67] | 1.06 [1.03, 1.09] | 1.00 [1.00, 1.01] | 0.15 [0.15, 0.16] | 0.15 [0.15, 0.16] | 1.59 [1.53, 1.67] |
| metaqa | 1.50 [1.49, 1.52] | 1.05 [1.04, 1.06] | 1.03 [1.02, 1.03] | 1.03 [1.02, 1.03] | 1.19 [1.17, 1.21] | 1.29 [1.28, 1.32] |
| musique | 1.61 [1.58, 1.63] | 1.09 [1.08, 1.10] | 1.07 [1.06, 1.09] | 1.07 [1.06, 1.09] | 2.32 [2.28, 2.40] | 0.75 [0.73, 0.75] |
| webqsp | 1.45 [1.43, 1.48] | 0.96 [0.93, 0.98] | 0.98 [0.96, 0.99] | 0.93 [0.90, 0.96] | 0.93 [0.90, 0.96] | 1.45 [1.43, 1.47] |
| hotpotqa | 1.04 [1.03, 1.05] | 1.01 [1.00, 1.01] | 1.00 [0.99, 1.02] | 0.93 [0.90, 0.94] | 0.93 [0.90, 0.94] | 1.04 [1.03, 1.05] |
| 2wiki | 1.03 [1.02, 1.04] | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.01] | 0.91 [0.89, 0.92] | 0.91 [0.89, 0.92] | 1.03 [1.02, 1.04] |

The serve path's stages (questions run on, p50 ms where run, share of serve), the walks each path ran, the serve path's extra walks, and each path's p50 by position:

- squad, fresh: bs 200, 0.068, 0.00; columns 200, 0.259, 0.01; fill 200, 0.169, 0.01; kind 200, 0.146, 0.01; link 200, 0.001, 0.00; order 200, 1.859, 0.09; pool 200, 0.081, 0.00; restarts 200, 0.242, 0.01; w_a1all 200, 10.696, 0.49; w_a1st 200, 8.561, 0.39
  walks run {'build': {'A1st': 0, 'A1all': 0, 'Kst': 0, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}}; extra {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}; by position build first 0.11, second 0.16; serve first 22.23, second 22.17
- squad, c12: bs 200, 0.068, 0.00; columns 200, 0.262, 0.01; fill 200, 0.166, 0.01; kind 200, 0.144, 0.01; link 200, 0.001, 0.00; order 200, 1.851, 0.07; pool 200, 0.078, 0.00; restarts 200, 0.236, 0.01; w_a1all 200, 13.162, 0.51; w_a1st 200, 9.609, 0.38
  walks run {'build': {'A1st': 0, 'A1all': 0, 'Kst': 0, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}}; extra {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}; by position build first 0.11, second 0.16; serve first 25.56, second 26.19
  P4b's filed build p50 (P4b's session): {'fresh': 0.11959999392274767, 'c12': 0.11405000986997038}; power {'start': {'plugged': True, 'percent': 99.0}, 'end': {'plugged': True, 'percent': 99.0}}; pinned True; index {'reference_s': 42.51786947250366, 'p4b_form_s': 0.8838415145874023, 'p4b_form_bytes': 15453444, 'jit_question_s': 8.814399719238281, 'setup_total_s': 2.9694273471832275}
- metaqa, fresh: bs 200, 0.084, 0.00; columns 200, 0.636, 0.01; fill 200, 0.653, 0.01; kind 200, 0.149, 0.00; link 200, 0.040, 0.00; order 200, 0.010, 0.00; pool 200, 0.247, 0.01; restarts 200, 0.563, 0.01; w_a1all 200, 15.279, 0.33; w_a1st 200, 13.989, 0.30; w_kst 200, 5.418, 0.11; w_ktall 200, 4.856, 0.11; w_ktst 200, 4.220, 0.10
  walks run {'build': {'A1st': 7, 'A1all': 0, 'Kst': 194, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 193, 'A1all': 200, 'Kst': 6, 'KTst': 200, 'KTall': 200}; by position build first 6.59, second 6.62; serve first 45.95, second 45.96
- metaqa, c12: bs 200, 0.084, 0.00; columns 200, 0.620, 0.01; fill 200, 0.450, 0.01; kind 200, 0.148, 0.00; link 200, 0.041, 0.00; order 200, 0.008, 0.00; pool 200, 0.246, 0.01; restarts 200, 0.558, 0.01; w_a1all 200, 15.376, 0.34; w_a1st 200, 13.944, 0.31; w_kst 200, 4.814, 0.11; w_ktall 200, 4.014, 0.10; w_ktst 200, 3.623, 0.09
  walks run {'build': {'A1st': 42, 'A1all': 0, 'Kst': 159, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 158, 'A1all': 200, 'Kst': 41, 'KTst': 200, 'KTall': 200}; by position build first 6.46, second 6.31; serve first 44.37, second 44.03
  P4b's filed build p50 (P4b's session): {'fresh': 6.360399987897836, 'c12': 6.291600002441555}; power {'start': {'plugged': True, 'percent': 99.0}, 'end': {'plugged': True, 'percent': 99.0}}; pinned True; index {'reference_s': 63.56863451004028, 'p4b_form_s': 0.865081787109375, 'p4b_form_bytes': 6981620, 'jit_question_s': 9.440654516220093, 'setup_total_s': 13.237035274505615}
- musique, fresh: bs 200, 0.097, 0.00; columns 200, 0.349, 0.01; fill 200, 0.657, 0.02; kind 200, 0.141, 0.00; link 200, 0.077, 0.00; order 200, 2.631, 0.07; pool 200, 0.330, 0.01; restarts 200, 0.534, 0.01; w_a1all 200, 11.237, 0.29; w_a1st 200, 11.465, 0.30; w_kst 138, 4.712, 0.08; w_ktall 138, 5.909, 0.11; w_ktst 138, 5.469, 0.10
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 138, 'KTall': 138}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 138, 'KTst': 138, 'KTall': 138}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 138, 'KTst': 0, 'KTall': 0}; by position build first 35.64, second 34.87; serve first 39.63, second 40.81
- musique, c12: bs 200, 0.096, 0.00; columns 200, 0.371, 0.01; fill 200, 0.665, 0.02; kind 200, 0.141, 0.00; link 200, 0.068, 0.00; order 200, 2.751, 0.07; pool 200, 0.331, 0.01; restarts 200, 0.546, 0.01; w_a1all 200, 11.220, 0.28; w_a1st 200, 11.212, 0.29; w_kst 154, 4.563, 0.09; w_ktall 154, 6.208, 0.12; w_ktst 154, 4.949, 0.10
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 154, 'KTall': 154}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 154, 'KTst': 154, 'KTall': 154}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 154, 'KTst': 0, 'KTall': 0}; by position build first 36.84, second 36.43; serve first 39.90, second 41.44
  P4b's filed build p50 (P4b's session): {'fresh': 34.39739999885205, 'c12': 35.125950002111495}; power {'start': {'plugged': True, 'percent': 99.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 70.07494831085205, 'p4b_form_s': 0.9682397842407227, 'p4b_form_bytes': 62243596, 'jit_question_s': 9.390833377838135, 'setup_total_s': 6.015597105026245}
- webqsp, fresh: bs 200, 0.044, 0.00; columns 200, 0.384, 0.01; fill 200, 0.742, 0.02; kind 200, 0.141, 0.00; link 200, 0.048, 0.00; order 200, 0.004, 0.00; pool 200, 0.279, 0.01; restarts 200, 0.529, 0.01; w_a1all 200, 10.023, 0.22; w_a1st 200, 12.858, 0.28; w_kst 192, 7.413, 0.17; w_ktall 192, 6.038, 0.14; w_ktst 192, 5.080, 0.14
  walks run {'build': {'A1st': 139, 'A1all': 5, 'Kst': 61, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 192, 'KTst': 192, 'KTall': 192}}; extra {'A1st': 61, 'A1all': 195, 'Kst': 131, 'KTst': 192, 'KTall': 192}; by position build first 12.74, second 13.26; serve first 42.53, second 43.48
- webqsp, c12: bs 200, 0.044, 0.00; columns 200, 0.381, 0.01; fill 200, 0.733, 0.02; kind 200, 0.137, 0.00; link 200, 0.048, 0.00; order 200, 0.004, 0.00; pool 200, 0.277, 0.01; restarts 200, 0.539, 0.01; w_a1all 200, 10.034, 0.23; w_a1st 200, 12.384, 0.28; w_kst 185, 7.210, 0.16; w_ktall 185, 5.885, 0.14; w_ktst 185, 5.218, 0.14
  walks run {'build': {'A1st': 146, 'A1all': 12, 'Kst': 55, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 185, 'KTst': 185, 'KTall': 185}}; extra {'A1st': 54, 'A1all': 188, 'Kst': 130, 'KTst': 185, 'KTall': 185}; by position build first 12.40, second 13.38; serve first 40.31, second 42.45
  P4b's filed build p50 (P4b's session): {'fresh': 13.196600004448555, 'c12': 13.212449994171038}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 109.61551976203918, 'p4b_form_s': 2.4248523712158203, 'p4b_form_bytes': 419497124, 'jit_question_s': 8.501378774642944, 'setup_total_s': 33.21898627281189}
- hotpotqa, fresh: bs 200, 0.069, 0.00; columns 200, 0.401, 0.01; fill 200, 0.469, 0.01; kind 200, 0.138, 0.00; link 200, 0.107, 0.00; order 200, 3.343, 0.06; pool 200, 0.096, 0.00; restarts 200, 0.555, 0.01; w_a1all 200, 15.332, 0.27; w_a1st 200, 15.748, 0.28; w_kst 181, 7.518, 0.13; w_ktall 181, 6.817, 0.11; w_ktst 181, 6.173, 0.12
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 181, 'KTall': 181}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 181, 'KTst': 181, 'KTall': 181}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 181, 'KTst': 0, 'KTall': 0}; by position build first 49.43, second 48.88; serve first 55.63, second 56.28
- hotpotqa, c12: bs 200, 0.068, 0.00; columns 200, 0.391, 0.01; fill 200, 0.464, 0.01; kind 200, 0.136, 0.00; link 200, 0.106, 0.00; order 200, 3.185, 0.06; pool 200, 0.096, 0.00; restarts 200, 0.554, 0.01; w_a1all 200, 14.945, 0.28; w_a1st 200, 15.482, 0.29; w_kst 178, 7.081, 0.12; w_ktall 178, 6.490, 0.11; w_ktst 178, 5.719, 0.11
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 178, 'KTall': 178}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 178, 'KTst': 178, 'KTall': 178}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 178, 'KTst': 0, 'KTall': 0}; by position build first 46.42, second 47.23; serve first 54.14, second 53.76
  P4b's filed build p50 (P4b's session): {'fresh': 47.721650000312366, 'c12': 47.27614999865182}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 165.52832984924316, 'p4b_form_s': 6.720921754837036, 'p4b_form_bytes': 1003101252, 'jit_question_s': 9.06233286857605, 'setup_total_s': 94.49372673034668}
- 2wiki, fresh: bs 200, 0.072, 0.00; columns 200, 0.464, 0.01; fill 200, 0.460, 0.01; kind 200, 0.137, 0.00; link 200, 0.072, 0.00; order 200, 3.482, 0.06; pool 200, 0.098, 0.00; restarts 200, 0.553, 0.01; w_a1all 200, 16.394, 0.27; w_a1st 200, 15.504, 0.26; w_kst 200, 7.443, 0.13; w_ktall 200, 7.250, 0.12; w_ktst 200, 6.796, 0.12
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 200, 'KTall': 200}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 200, 'KTst': 0, 'KTall': 0}; by position build first 52.38, second 52.29; serve first 58.21, second 59.90
- 2wiki, c12: bs 200, 0.072, 0.00; columns 200, 0.451, 0.01; fill 200, 0.462, 0.01; kind 200, 0.136, 0.00; link 200, 0.069, 0.00; order 200, 3.446, 0.06; pool 200, 0.099, 0.00; restarts 200, 0.551, 0.01; w_a1all 200, 16.397, 0.28; w_a1st 200, 15.825, 0.26; w_kst 200, 7.451, 0.13; w_ktall 200, 7.166, 0.12; w_ktst 200, 7.072, 0.13
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 200, 'KTall': 200}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 200, 'KTst': 0, 'KTall': 0}; by position build first 52.17, second 52.25; serve first 61.10, second 60.60
  P4b's filed build p50 (P4b's session): {'fresh': 50.301299983402714, 'c12': 50.80389999784529}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 180.7642641067505, 'p4b_form_s': 9.133471965789795, 'p4b_form_bytes': 1408798196, 'jit_question_s': 8.893442630767822, 'setup_total_s': 119.23871183395386}

Checks:

- squad, fresh: kept 200/200; every check held on every question
- squad, c12: kept 200/200; every check held on every question
- metaqa, fresh: kept 200/200; every check held on every question
- metaqa, c12: kept 200/200; every check held on every question
- musique, fresh: kept 200/200; every check held on every question
- musique, c12: kept 200/200; every check held on every question
- webqsp, fresh: kept 200/200; every check held on every question
- webqsp, c12: kept 200/200; every check held on every question
- hotpotqa, fresh: kept 200/200; every check held on every question
- hotpotqa, c12: kept 200/200; every check held on every question
- 2wiki, fresh: kept 200/200; every check held on every question
- 2wiki, c12: kept 200/200; every check held on every question
