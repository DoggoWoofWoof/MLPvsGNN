P4c: the KT input's serving path beside KIND_T's build, both on every question, cold, batch 1, one thread, P4b's walks (p50 ms; ratios serve / build with 2,000-resample paired bootstrap 95% CIs). Call: **TIMED**.

The fresh set (questions 201-400):

| dataset | kept | build | serve | head | serve / build | serve + head / build | extra p50 | rows p50 | B_q<=0 | entity | CPU load before | free GB before | page faults |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 200/200 | 0.15 | 21.72 | 0.53 | 148.77 [143.91, 184.92] | 152.96 [147.87, 189.51] | 21.61 | 50 | 200 | 0 | 13.3 | 5.98 | 169 |
| metaqa | 200/200 | 6.49 | 45.38 | 0.67 | 6.99 [6.86, 7.16] | 7.10 [6.97, 7.28] | 39.13 | 2036 | 0 | 200 | 10.1 | 5.8 | 9296 |
| musique | 200/200 | 35.45 | 39.33 | 0.74 | 1.11 [1.07, 1.15] | 1.13 [1.10, 1.18] | 3.98 | 2136 | 0 | 0 | 10.5 | 5.88 | 13095 |
| webqsp | 200/200 | 13.02 | 42.81 | 0.73 | 3.29 [3.11, 3.52] | 3.36 [3.16, 3.58] | 31.52 | 2126 | 0 | 195 | 6.8 | 4.71 | 12513 |
| hotpotqa | 200/200 | 47.40 | 55.59 | 0.57 | 1.17 [1.14, 1.19] | 1.19 [1.15, 1.20] | 7.35 | 90 | 0 | 0 | 7.3 | 3.19 | 9297 |
| 2wiki | 200/200 | 50.88 | 58.04 | 0.58 | 1.14 [1.13, 1.16] | 1.15 [1.14, 1.17] | 7.44 | 105 | 0 | 0 | 5.4 | 2.26 | 6195 |

The c12 set (questions 1-200):

| dataset | kept | build | serve | head | serve / build | serve + head / build | extra p50 | rows p50 | B_q<=0 | entity | CPU load before | free GB before | page faults |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 200/200 | 0.15 | 25.64 | 0.53 | 172.21 [168.20, 208.19] | 175.76 [171.62, 212.45] | 25.49 | 50 | 200 | 0 | 6.9 | 6.0 | 123 |
| metaqa | 200/200 | 6.33 | 43.42 | 0.66 | 6.86 [6.55, 7.14] | 6.97 [6.64, 7.26] | 36.77 | 2038 | 0 | 200 | 7.1 | 5.94 | 10114 |
| musique | 200/200 | 36.16 | 40.43 | 0.66 | 1.12 [1.10, 1.14] | 1.14 [1.12, 1.16] | 4.40 | 2142 | 0 | 0 | 12.1 | 5.83 | 8605 |
| webqsp | 200/200 | 13.01 | 41.53 | 0.67 | 3.19 [3.00, 3.38] | 3.25 [3.06, 3.44] | 30.18 | 2122 | 0 | 188 | 10.4 | 4.65 | 7766 |
| hotpotqa | 200/200 | 46.57 | 53.61 | 0.55 | 1.15 [1.13, 1.17] | 1.16 [1.14, 1.18] | 6.75 | 89 | 0 | 0 | 12.2 | 3.14 | 1235 |
| 2wiki | 200/200 | 51.54 | 58.83 | 0.55 | 1.14 [1.12, 1.18] | 1.15 [1.13, 1.19] | 6.96 | 103 | 0 | 0 | 3.4 | 2.24 | 1819 |

End to end on C12's set, cold p50 ms: C12's model stages (C12's session, today's pools) plus the build timed here (umlp = twin0, gnn = gnn0, zrc, zsp) or plus the serve path and the head (zrck on zrc's stages, zspk on zsp's):

| dataset | umlp | gnn | zrc | zsp | zrck | zspk | fastest GNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| squad | 2.73 | 4.33 | 1.72 | 1.82 | 27.92 | 28.05 | gnn |
| metaqa | 55.29 | 83.17 | 32.29 | 33.95 | 69.42 | 71.19 | zspk |
| musique | 110.64 | 177.56 | 71.21 | 77.44 | 75.78 | 81.70 | zspk |
| webqsp | 83.00 | 120.14 | 96.29 | 92.57 | 128.99 | 125.80 | gnn |
| hotpotqa | 50.84 | 52.86 | 48.95 | 49.36 | 56.45 | 56.75 | gnn |
| 2wiki | 55.40 | 56.92 | 53.75 | 53.87 | 62.26 | 62.13 | gnn |

Ratios on C12's set (paired bootstrap 95% CIs):

| dataset | universal | lean_on_build | lean_kt | fastest_gnn_over_zrck | universal_gnn_over_zrck | fastest_gnn_over_universal_mlp |
| --- | --- | --- | --- | --- | --- | --- |
| squad | 1.59 [1.54, 1.67] | 1.06 [1.03, 1.10] | 1.00 [1.00, 1.01] | 0.16 [0.15, 0.17] | 0.16 [0.15, 0.16] | 1.59 [1.53, 1.67] |
| metaqa | 1.50 [1.49, 1.51] | 1.05 [1.05, 1.07] | 1.03 [1.02, 1.03] | 1.03 [1.02, 1.03] | 1.20 [1.18, 1.21] | 1.29 [1.27, 1.30] |
| musique | 1.60 [1.58, 1.63] | 1.09 [1.08, 1.10] | 1.08 [1.07, 1.09] | 1.08 [1.07, 1.09] | 2.34 [2.28, 2.41] | 0.74 [0.73, 0.75] |
| webqsp | 1.45 [1.43, 1.47] | 0.96 [0.93, 0.97] | 0.98 [0.95, 0.99] | 0.93 [0.89, 0.96] | 0.93 [0.89, 0.96] | 1.45 [1.43, 1.47] |
| hotpotqa | 1.04 [1.03, 1.05] | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] | 0.94 [0.91, 0.95] | 0.94 [0.91, 0.95] | 1.04 [1.03, 1.05] |
| 2wiki | 1.03 [1.02, 1.04] | 1.00 [1.00, 1.01] | 1.00 [0.99, 1.01] | 0.91 [0.90, 0.94] | 0.91 [0.90, 0.94] | 1.03 [1.02, 1.04] |

The serve path's stages (questions run on, p50 ms where run, share of serve), the walks each path ran, the serve path's extra walks, and each path's p50 by position:

- squad, fresh: bs 200, 0.064, 0.00; columns 200, 0.256, 0.01; fill 200, 0.158, 0.01; kind 200, 0.135, 0.01; link 200, 0.001, 0.00; order 200, 1.826, 0.08; pool 200, 0.074, 0.00; restarts 200, 0.216, 0.01; w_a1all 200, 10.519, 0.49; w_a1st 200, 8.451, 0.39
  walks run {'build': {'A1st': 0, 'A1all': 0, 'Kst': 0, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}}; extra {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}; by position build first 0.11, second 0.16; serve first 21.87, second 21.70
- squad, c12: bs 200, 0.066, 0.00; columns 200, 0.260, 0.01; fill 200, 0.163, 0.01; kind 200, 0.138, 0.01; link 200, 0.001, 0.00; order 200, 1.839, 0.07; pool 200, 0.079, 0.00; restarts 200, 0.218, 0.01; w_a1all 200, 13.181, 0.51; w_a1st 200, 9.738, 0.38
  walks run {'build': {'A1st': 0, 'A1all': 0, 'Kst': 0, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}}; extra {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 0, 'KTall': 0}; by position build first 0.11, second 0.16; serve first 25.44, second 26.11
  P4b's filed build p50 (P4b's session): {'fresh': 0.11959999392274767, 'c12': 0.11405000986997038}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 41.39129185676575, 'p4b_form_s': 0.7880845069885254, 'p4b_form_bytes': 15453444, 'jit_question_s': 8.775283575057983, 'setup_total_s': 5.210585117340088}
- metaqa, fresh: bs 200, 0.082, 0.00; columns 200, 0.635, 0.01; fill 200, 0.651, 0.01; kind 200, 0.146, 0.00; link 200, 0.039, 0.00; order 200, 0.009, 0.00; pool 200, 0.245, 0.01; restarts 200, 0.556, 0.01; w_a1all 200, 15.189, 0.33; w_a1st 200, 13.868, 0.30; w_kst 200, 5.405, 0.11; w_ktall 200, 4.812, 0.11; w_ktst 200, 4.183, 0.10
  walks run {'build': {'A1st': 7, 'A1all': 0, 'Kst': 194, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 193, 'A1all': 200, 'Kst': 6, 'KTst': 200, 'KTall': 200}; by position build first 6.47, second 6.49; serve first 45.10, second 45.56
- metaqa, c12: bs 200, 0.082, 0.00; columns 200, 0.617, 0.01; fill 200, 0.425, 0.01; kind 200, 0.145, 0.00; link 200, 0.040, 0.00; order 200, 0.007, 0.00; pool 200, 0.240, 0.01; restarts 200, 0.551, 0.01; w_a1all 200, 15.114, 0.34; w_a1st 200, 13.696, 0.32; w_kst 200, 4.911, 0.11; w_ktall 200, 3.973, 0.10; w_ktst 200, 3.698, 0.09
  walks run {'build': {'A1st': 42, 'A1all': 0, 'Kst': 159, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 158, 'A1all': 200, 'Kst': 41, 'KTst': 200, 'KTall': 200}; by position build first 6.33, second 6.28; serve first 43.41, second 43.48
  P4b's filed build p50 (P4b's session): {'fresh': 6.360399987897836, 'c12': 6.291600002441555}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 60.20006561279297, 'p4b_form_s': 0.8104493618011475, 'p4b_form_bytes': 6981620, 'jit_question_s': 8.794357061386108, 'setup_total_s': 22.52598524093628}
- musique, fresh: bs 200, 0.093, 0.00; columns 200, 0.345, 0.01; fill 200, 0.648, 0.02; kind 200, 0.137, 0.00; link 200, 0.074, 0.00; order 200, 2.651, 0.07; pool 200, 0.323, 0.01; restarts 200, 0.530, 0.01; w_a1all 200, 11.156, 0.29; w_a1st 200, 11.290, 0.30; w_kst 138, 4.754, 0.08; w_ktall 138, 5.831, 0.11; w_ktst 138, 5.443, 0.10
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 138, 'KTall': 138}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 138, 'KTst': 138, 'KTall': 138}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 138, 'KTst': 0, 'KTall': 0}; by position build first 35.75, second 35.22; serve first 39.20, second 39.44
- musique, c12: bs 200, 0.093, 0.00; columns 200, 0.363, 0.01; fill 200, 0.650, 0.02; kind 200, 0.136, 0.00; link 200, 0.065, 0.00; order 200, 2.721, 0.07; pool 200, 0.322, 0.01; restarts 200, 0.533, 0.01; w_a1all 200, 11.154, 0.28; w_a1st 200, 11.300, 0.29; w_kst 154, 4.617, 0.09; w_ktall 154, 5.925, 0.12; w_ktst 154, 4.916, 0.10
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 154, 'KTall': 154}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 154, 'KTst': 154, 'KTall': 154}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 154, 'KTst': 0, 'KTall': 0}; by position build first 36.23, second 35.47; serve first 39.85, second 40.94
  P4b's filed build p50 (P4b's session): {'fresh': 34.39739999885205, 'c12': 35.125950002111495}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 62.412227392196655, 'p4b_form_s': 0.8823518753051758, 'p4b_form_bytes': 62243596, 'jit_question_s': 8.459795951843262, 'setup_total_s': 5.623216152191162}
- webqsp, fresh: bs 200, 0.044, 0.00; columns 200, 0.390, 0.01; fill 200, 0.740, 0.02; kind 200, 0.142, 0.00; link 200, 0.048, 0.00; order 200, 0.004, 0.00; pool 200, 0.279, 0.01; restarts 200, 0.533, 0.01; w_a1all 200, 10.047, 0.22; w_a1st 200, 12.858, 0.29; w_kst 192, 7.570, 0.16; w_ktall 192, 6.076, 0.14; w_ktst 192, 5.005, 0.14
  walks run {'build': {'A1st': 139, 'A1all': 5, 'Kst': 61, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 192, 'KTst': 192, 'KTall': 192}}; extra {'A1st': 61, 'A1all': 195, 'Kst': 131, 'KTst': 192, 'KTall': 192}; by position build first 12.93, second 13.39; serve first 42.81, second 42.88
- webqsp, c12: bs 200, 0.043, 0.00; columns 200, 0.385, 0.01; fill 200, 0.739, 0.02; kind 200, 0.139, 0.00; link 200, 0.049, 0.00; order 200, 0.004, 0.00; pool 200, 0.279, 0.01; restarts 200, 0.535, 0.01; w_a1all 200, 10.120, 0.23; w_a1st 200, 12.434, 0.28; w_kst 185, 7.221, 0.16; w_ktall 185, 6.068, 0.14; w_ktst 185, 5.302, 0.14
  walks run {'build': {'A1st': 146, 'A1all': 12, 'Kst': 55, 'KTst': 0, 'KTall': 0}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 185, 'KTst': 185, 'KTall': 185}}; extra {'A1st': 54, 'A1all': 188, 'Kst': 130, 'KTst': 185, 'KTall': 185}; by position build first 12.65, second 13.37; serve first 40.84, second 43.33
  P4b's filed build p50 (P4b's session): {'fresh': 13.196600004448555, 'c12': 13.212449994171038}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 105.92472791671753, 'p4b_form_s': 2.42086124420166, 'p4b_form_bytes': 419497124, 'jit_question_s': 8.452273607254028, 'setup_total_s': 32.86408448219299}
- hotpotqa, fresh: bs 200, 0.071, 0.00; columns 200, 0.399, 0.01; fill 200, 0.467, 0.01; kind 200, 0.140, 0.00; link 200, 0.107, 0.00; order 200, 3.296, 0.06; pool 200, 0.098, 0.00; restarts 200, 0.557, 0.01; w_a1all 200, 15.124, 0.27; w_a1st 200, 15.474, 0.28; w_kst 181, 7.427, 0.13; w_ktall 181, 6.503, 0.11; w_ktst 181, 6.039, 0.12
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 181, 'KTall': 181}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 181, 'KTst': 181, 'KTall': 181}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 181, 'KTst': 0, 'KTall': 0}; by position build first 48.06, second 47.25; serve first 54.68, second 55.98
- hotpotqa, c12: bs 200, 0.069, 0.00; columns 200, 0.393, 0.01; fill 200, 0.465, 0.01; kind 200, 0.140, 0.00; link 200, 0.108, 0.00; order 200, 3.246, 0.06; pool 200, 0.098, 0.00; restarts 200, 0.563, 0.01; w_a1all 200, 14.923, 0.28; w_a1st 200, 15.457, 0.29; w_kst 178, 7.166, 0.12; w_ktall 178, 6.350, 0.11; w_ktst 178, 5.634, 0.11
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 178, 'KTall': 178}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 178, 'KTst': 178, 'KTall': 178}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 178, 'KTst': 0, 'KTall': 0}; by position build first 46.13, second 47.01; serve first 53.32, second 54.36
  P4b's filed build p50 (P4b's session): {'fresh': 47.721650000312366, 'c12': 47.27614999865182}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 159.62440848350525, 'p4b_form_s': 6.655618190765381, 'p4b_form_bytes': 1003101252, 'jit_question_s': 8.909655570983887, 'setup_total_s': 93.59261870384216}
- 2wiki, fresh: bs 200, 0.070, 0.00; columns 200, 0.459, 0.01; fill 200, 0.462, 0.01; kind 200, 0.136, 0.00; link 200, 0.071, 0.00; order 200, 3.461, 0.06; pool 200, 0.098, 0.00; restarts 200, 0.561, 0.01; w_a1all 200, 16.094, 0.28; w_a1st 200, 15.211, 0.26; w_kst 200, 7.126, 0.13; w_ktall 200, 7.111, 0.12; w_ktst 200, 6.635, 0.12
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 200, 'KTall': 200}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 200, 'KTst': 0, 'KTall': 0}; by position build first 50.90, second 50.83; serve first 57.47, second 58.22
- 2wiki, c12: bs 200, 0.069, 0.00; columns 200, 0.451, 0.01; fill 200, 0.462, 0.01; kind 200, 0.134, 0.00; link 200, 0.069, 0.00; order 200, 3.403, 0.06; pool 200, 0.099, 0.00; restarts 200, 0.553, 0.01; w_a1all 200, 16.011, 0.27; w_a1st 200, 15.392, 0.27; w_kst 200, 7.400, 0.12; w_ktall 200, 7.044, 0.12; w_ktst 200, 6.942, 0.12
  walks run {'build': {'A1st': 200, 'A1all': 200, 'Kst': 0, 'KTst': 200, 'KTall': 200}, 'serve': {'A1st': 200, 'A1all': 200, 'Kst': 200, 'KTst': 200, 'KTall': 200}}; extra {'A1st': 0, 'A1all': 0, 'Kst': 200, 'KTst': 0, 'KTall': 0}; by position build first 51.14, second 51.75; serve first 59.34, second 58.15
  P4b's filed build p50 (P4b's session): {'fresh': 50.301299983402714, 'c12': 50.80389999784529}; power {'start': {'plugged': True, 'percent': 100.0}, 'end': {'plugged': True, 'percent': 100.0}}; pinned True; index {'reference_s': 180.7898907661438, 'p4b_form_s': 9.145944118499756, 'p4b_form_bytes': 1408798196, 'jit_question_s': 9.002624273300171, 'setup_total_s': 120.07939434051514}

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
