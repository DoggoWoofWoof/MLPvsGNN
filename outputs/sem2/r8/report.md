# SEM2 R8 part A: the room

Declared in docs/SEM2_EDGE_OPERATORS.md section 16. Questions: the selection carve; F = 1000, seeds 25, two hops. H(M') in points: the first stage's ALL-gold coverage gain if a perfect router spent M' added nodes.

| dataset | questions | cov0 | missed golds: class 1 / class 2 / beyond (%) | H(25) | H(100) | H(500) | H(1500) [95%] | ROOM | median cost* | median unordered cost |
|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| multihoprag | 224 | 100.00 | 0.0 / 0.0 / 0.0 | 0.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | NO_ROOM | None | None |
| squad | 1498 | 99.60 | 33.3 / 50.0 / 16.7 | 0.33 | 0.33 | 0.33 | 0.33 [0.07, 0.67] | NO_ROOM | 2.0 | 275857.0 |
| musique | 1534 | 65.38 | 64.1 / 31.1 / 4.8 | 32.79 | 32.79 | 32.79 | 32.79 [30.64, 35.20] | ROOM | 1.0 | 527825.0 |
| metaqa | 1497 | 18.70 | 14.6 / 53.9 / 31.5 | 63.73 | 66.33 | 66.40 | 66.40 [63.99, 68.81] | ROOM | 2.0 | 46924.0 |
| webqsp | 1507 | 25.15 | 83.5 / 15.6 / 0.9 | 70.07 | 73.46 | 73.52 | 73.52 [71.33, 75.71] | ROOM | 2.0 | 166404092.5 |
| hotpotqa | 1508 | 83.82 | 95.1 / 4.0 / 0.8 | 16.05 | 16.05 | 16.05 | 16.05 [14.19, 17.90] | ROOM | 1.0 | 3423191.5 |
| 2wiki | 1496 | 47.66 | 98.3 / 1.6 / 0.1 | 52.27 | 52.27 | 52.27 | 52.27 [49.67, 54.68] | ROOM | 1.0 | 8863648.0 |

ROOM on: musique, metaqa, webqsp, hotpotqa, 2wiki. Missing rows: none.
