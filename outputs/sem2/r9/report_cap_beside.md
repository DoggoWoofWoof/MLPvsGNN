# SEM2 R9 part A9: the reads beside the ladder (17.11)

Read-only on the cap records; nothing solved; the paired reads are on a selected subset (the questions both programs were proven on) and decide nothing.

## Proven shares per rung, pooled over both scorers (the declared reading) and per scorer

| dataset | rung | unproven instances | pooled share | pooled reads | W0 share | W1 share | cells that flip |
|---|---|---:|---:|---|---:|---:|---|
| musique | 100/50 | 0 | 1.0000 | True | 1.0000 | 1.0000 | - |
| musique | 300/150 | 0 | 1.0000 | True | 1.0000 | 1.0000 | - |
| musique | 1000/500 | 11 | 0.9127 | False | 0.9683 | 0.8571 | - |
| metaqa | 100/50 | 0 | 1.0000 | True | 1.0000 | 1.0000 | - |
| metaqa | 300/150 | 0 | 1.0000 | True | 1.0000 | 1.0000 | - |
| metaqa | 1000/500 | 1 | 0.9933 | False | 0.9867 | 1.0000 | W1 |
| webqsp | 100/50 | 0 | 1.0000 | True | 1.0000 | 1.0000 | - |
| webqsp | 300/150 | 2 | 0.9867 | False | 1.0000 | 0.9733 | W0 |
| webqsp | 1000/500 | 15 | 0.9000 | False | 0.9467 | 0.8533 | - |
| hotpotqa | 100/50 | 0 | 1.0000 | True | 1.0000 | 1.0000 | - |
| hotpotqa | 300/150 | 1 | 0.9839 | False | 1.0000 | 0.9677 | W0 |
| hotpotqa | 1000/500 | 9 | 0.8548 | False | 0.8065 | 0.9032 | - |

## Completion against the rung 100/50, on the questions proven at both (n of the slice)

| dataset | scorer | rung | n | C1 on the subset % | rung on the subset % | difference (points) | interval | call | recall gain, all questions (points) |
|---|---|---|---:|---:|---:|---:|---|---|---:|
| musique | W0 | 300/150 | 63 of 63 | 26.98 | 26.98 | 0.0 | [-11.11, 11.11] | LEVEL | 28.57 |
| musique | W0 | 1000/500 | 61 of 63 | 26.23 | 24.59 | -1.64 | [-13.11, 9.84] | LEVEL | 63.49 |
| musique | W1 | 300/150 | 63 of 63 | 33.33 | 25.4 | -7.94 | [-17.46, 0.0] | LEVEL | 25.4 |
| musique | W1 | 1000/500 | 54 of 63 | 33.33 | 20.37 | -12.96 | [-25.93, 0.0] | LEVEL | 57.14 |
| metaqa | W0 | 300/150 | 75 of 75 | 42.67 | 25.33 | -17.33 | [-26.67, -8.0] | BELOW | 38.67 |
| metaqa | W0 | 1000/500 | 74 of 75 | 43.24 | 25.68 | -17.57 | [-27.03, -8.11] | BELOW | 42.67 |
| metaqa | W1 | 300/150 | 75 of 75 | 21.33 | 12.0 | -9.33 | [-17.33, -2.67] | BELOW | 45.33 |
| metaqa | W1 | 1000/500 | 75 of 75 | 21.33 | 12.0 | -9.33 | [-17.33, -2.67] | BELOW | 49.33 |
| webqsp | W0 | 300/150 | 75 of 75 | 36.0 | 34.67 | -1.33 | [-8.0, 5.33] | LEVEL | 12.0 |
| webqsp | W0 | 1000/500 | 71 of 75 | 36.62 | 26.76 | -9.86 | [-18.31, -1.41] | BELOW | 20.0 |
| webqsp | W1 | 300/150 | 73 of 75 | 24.66 | 17.81 | -6.85 | [-13.7, 0.0] | LEVEL | 9.33 |
| webqsp | W1 | 1000/500 | 64 of 75 | 23.44 | 14.06 | -9.38 | [-18.75, 0.0] | LEVEL | 22.67 |
| hotpotqa | W0 | 300/150 | 31 of 31 | 45.16 | 41.94 | -3.23 | [-22.58, 16.13] | LEVEL | 45.16 |
| hotpotqa | W0 | 1000/500 | 25 of 31 | 44.0 | 12.0 | -32.0 | [-52.0, -12.0] | BELOW | 54.84 |
| hotpotqa | W1 | 300/150 | 30 of 31 | 43.33 | 33.33 | -10.0 | [-30.0, 10.0] | LEVEL | 51.61 |
| hotpotqa | W1 | 1000/500 | 28 of 31 | 39.29 | 17.86 | -21.43 | [-39.29, -3.57] | BELOW | 54.84 |

## W0 minus W1 at each rung, on the questions proven in both

| dataset | rung | n | completion W0 - W1 (points) | interval | call | recall W0 - W1 (points) |
|---|---|---:|---:|---|---|---:|
| musique | 100/50 | 63 | -6.35 | [-20.63, 7.94] | LEVEL | -7.94 |
| musique | 300/150 | 63 | 1.59 | [-12.7, 15.87] | LEVEL | -4.76 |
| musique | 1000/500 | 52 | 7.69 | [-9.62, 25.0] | LEVEL | -1.59 |
| metaqa | 100/50 | 75 | 21.33 | [8.0, 36.0] | ABOVE | 8.0 |
| metaqa | 300/150 | 75 | 13.33 | [1.33, 26.67] | ABOVE | 1.33 |
| metaqa | 1000/500 | 74 | 14.86 | [2.7, 27.03] | ABOVE | 1.33 |
| webqsp | 100/50 | 75 | 12.0 | [0.0, 24.0] | LEVEL | 5.33 |
| webqsp | 300/150 | 73 | 17.81 | [5.48, 30.14] | ABOVE | 8.0 |
| webqsp | 1000/500 | 60 | 11.67 | [0.0, 23.33] | LEVEL | 2.67 |
| hotpotqa | 100/50 | 31 | 3.23 | [-16.13, 22.58] | LEVEL | 3.23 |
| hotpotqa | 300/150 | 30 | 6.67 | [-13.33, 26.67] | LEVEL | -3.23 |
| hotpotqa | 1000/500 | 24 | -4.17 | [-16.67, 8.33] | LEVEL | 3.23 |

Missing: 2wiki.
