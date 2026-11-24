# SEM2 R1k split: the ceiling over seen and unseen targets (post hoc)

Declared in docs/SEM2_EDGE_OPERATORS.md section 8.12; decides nothing (8.11's verdict stands). MRR_T in points; C = ceiling - rotation.

| dataset | subset | targets % | rotation | ceiling | C |
| --- | --- | --- | --- | --- | --- |
| metaqa | all | 100.0 | 10.23 | 11.13 | +0.90 [+0.78, +1.02] |
| metaqa | seen target (train node) | 79.89 | 10.77 | 11.47 | +0.71 [+0.59, +0.83] |
| metaqa | unseen target (test or dev node) | 20.11 | 8.09 | 9.75 | +1.67 [+1.32, +2.08] |
| metaqa | seen, degree > 0 | 63.85 | 12.24 | 13.01 | +0.77 [+0.64, +0.91] |
| metaqa | seen, degree 0 | 16.04 | 4.88 | 5.36 | +0.48 [+0.24, +0.76] |
| multihoprag | all | 100.0 | 23.43 | 28.35 | +4.93 [+4.06, +5.81] |
| multihoprag | seen target (train node) | 80.13 | 23.71 | 27.74 | +4.03 [+3.17, +5.05] |
| multihoprag | unseen target (test or dev node) | 19.87 | 22.3 | 30.82 | +8.52 [+6.23, +11.00] |
| multihoprag | seen, degree > 0 | 80.13 | 23.71 | 27.74 | +4.03 [+3.17, +5.05] |
| musique | all | 100.0 | 12.1 | 14.27 | +2.16 [+1.85, +2.50] |
| musique | seen target (train node) | 79.6 | 12.86 | 14.19 | +1.33 [+1.09, +1.63] |
| musique | unseen target (test or dev node) | 20.4 | 9.16 | 14.57 | +5.41 [+4.31, +6.55] |
| musique | seen, degree > 0 | 73.67 | 13.15 | 14.35 | +1.19 [+0.93, +1.47] |
| musique | seen, degree 0 | 5.93 | 9.16 | 12.23 | +3.07 [+1.56, +4.73] |
| squad | all | 100.0 | 27.9 | 32.76 | +4.86 [+4.33, +5.43] |
| squad | seen target (train node) | 79.65 | 28.85 | 32.07 | +3.21 [+2.73, +3.73] |
| squad | unseen target (test or dev node) | 20.35 | 24.15 | 35.47 | +11.32 [+9.52, +13.20] |
| squad | seen, degree > 0 | 79.51 | 28.89 | 32.1 | +3.21 [+2.74, +3.72] |
| squad | seen, degree 0 | 0.14 | 6.8 | 11.52 | +4.72 [+0.03, +13.33] |
