# UL4 report (docs/UL4_UNIVERSAL_LP.md)


## Layer L: the shared link keep-rule (dev solves, eval grades the integer rule; point values)

### Key space K1 (15 keys)

| program | rho | worst price tau* | largest own optimum | cost of universality | binding reach cells |
|---|---|---|---|---|---|
| S protected (A.1, the call) | 0.80 | 90.6 | 80.6 | 10.0 | musique/RK/CH, hotpotqa/RK/GP |
| S protected (A.1, the call) | 0.90 | 94.1 | 81.9 | 12.2 | musique/RK/CH |
| S protected (A.1, the call) | 0.95 | 97.0 | 82.6 | 14.4 | musique/RK/CH |
| S protected (A.1, the call) | 0.99 | 99.3 | 95.4 | 4.0 | musique/RK/CH, metaqa/SRK/GP, hotpotqa/RK/GP |
| S free | 0.80 | 74.0 | 49.6 | 24.4 | musique/RK/CH, 2wiki/SRK/GP |
| S free | 0.90 | 85.6 | 63.3 | 22.2 | musique/RK/CH, hotpotqa/SRK/GP, 2wiki/SRK/GP |
| S free | 0.95 | 92.7 | 78.7 | 14.0 | musique/RK/CH, hotpotqa/SRK/GP, 2wiki/SRK/GP |
| S free | 0.99 | 98.5 | 95.4 | 3.1 | musique/RK/CH, metaqa/SRK/GP, hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/SRK/GP |

The price of one point of reach near the call (tau*(0.96) - tau*(0.94)) / 2, in points of price: S protected 0.6; S free 1.4.

**Each judged cell's own optimum at rho 0.95 (dev price; the least a rule made for that cell alone could pay), and the S floor (the price of S alone):**

| judged cell | own optimum, S free | own optimum, S protected | S floor (dev) | S floor (eval) | S-alone reach (dev) |
|---|---|---|---|---|---|
| multihoprag/RK/GP | 41.0 | 41.0 | 0.0 | 0.0 | 0.0 |
| musique/RK/GP | 78.5 | 78.5 | 0.0 | 0.0 | 0.0 |
| musique/RK/CH | 78.7 | 78.7 | 0.0 | 0.0 | 0.0 |
| metaqa/SRK/GP | 17.1 | 82.6 | 71.6 | 72.3 | 4.1 |
| webqsp/SRK/GP | 25.1 | 52.3 | 36.9 | 38.5 | 36.2 |
| hotpotqa/RK/GP | 74.8 | 74.8 | 0.0 | 0.0 | 0.0 |
| hotpotqa/SRK/GP | 46.7 | 53.8 | 53.8 | 52.1 | 99.7 |
| 2wiki/RK/GP | 58.5 | 58.5 | 0.0 | 0.0 | 0.0 |
| 2wiki/SRK/GP | 43.0 | 43.0 | 40.0 | 35.3 | 93.8 |

**The cost in reach of a price (max over x of the worst judged cell's dev reach, at price <= tau for every cell):**

| price cap tau | S protected | S free |
|---|---|---|
| 0.30 | infeasible | 35.1 |
| 0.40 | infeasible | 46.7 |
| 0.50 | infeasible | 57.7 |
| 0.60 | infeasible | 67.2 |
| 0.80 | 42.9 | 85.3 |

**The integer rule, S protected (the call), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 97.5; kept 12 of 15 keys; dropped: K|0, K|2, R|0).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 82.9 | 99.9 / 83.8 | OUT |
| musique/RK/GP | 95.4 / 81.6 | 97.2 / 86.4 | OUT |
| musique/RK/CH | 95.3 / 81.6 | 97.2 / 86.4 | OUT |
| metaqa/SRK/GP | 95.3 / 97.5 | 95.6 / 97.5 | OUT |
| webqsp/SRK/GP | 99.3 / 89.1 | 99.8 / 90.6 | OUT |
| hotpotqa/RK/GP | 95.8 / 79.4 | 96.2 / 62.6 | OUT |
| hotpotqa/SRK/GP | 99.9 / 90.2 | 100.0 / 80.9 | OUT |
| 2wiki/RK/GP | 99.8 / 79.9 | 99.8 / 78.8 | OUT |
| 2wiki/SRK/GP | 99.9 / 87.1 | 99.9 / 86.6 | OUT |

**The integer rule, S free (beside), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 97.5; kept 12 of 15 keys; dropped: K|0, K|2, R|0).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 82.9 | 99.9 / 83.8 | OUT |
| musique/RK/GP | 95.4 / 81.6 | 97.2 / 86.4 | OUT |
| musique/RK/CH | 95.3 / 81.6 | 97.2 / 86.4 | OUT |
| metaqa/SRK/GP | 95.3 / 97.5 | 95.6 / 97.5 | OUT |
| webqsp/SRK/GP | 99.3 / 89.1 | 99.8 / 90.6 | OUT |
| hotpotqa/RK/GP | 95.8 / 79.4 | 96.2 / 62.6 | OUT |
| hotpotqa/SRK/GP | 99.9 / 90.2 | 100.0 / 80.9 | OUT |
| 2wiki/RK/GP | 99.8 / 79.9 | 99.8 / 78.8 | OUT |
| 2wiki/SRK/GP | 99.9 / 87.1 | 99.9 / 86.6 | OUT |

### Key space K2 (115 keys)

| program | rho | worst price tau* | largest own optimum | cost of universality | binding reach cells |
|---|---|---|---|---|---|
| S protected (A.1, the call) | 0.80 | 84.1 | 80.9 | 3.1 | hotpotqa/RK/GP, 2wiki/RK/GP |
| S protected (A.1, the call) | 0.90 | 91.1 | 82.4 | 8.7 | musique/RK/GP, hotpotqa/RK/GP |
| S protected (A.1, the call) | 0.95 | 95.0 | 83.1 | 12.0 | musique/RK/GP, hotpotqa/RK/GP |
| S protected (A.1, the call) | 0.99 | 98.8 | 88.8 | 10.0 | musique/RK/GP, hotpotqa/RK/GP |
| S free | 0.80 | 55.1 | 34.4 | 20.7 | musique/RK/CH |
| S free | 0.90 | 73.6 | 55.4 | 18.2 | musique/RK/CH, webqsp/SRK/GP, hotpotqa/SRK/GP, 2wiki/SRK/GP |
| S free | 0.95 | 85.8 | 71.5 | 14.3 | musique/RK/CH, webqsp/SRK/GP, 2wiki/SRK/GP |
| S free | 0.99 | 96.9 | 88.6 | 8.3 | musique/RK/GP, webqsp/SRK/GP, hotpotqa/RK/GP, 2wiki/SRK/GP |

The price of one point of reach near the call (tau*(0.96) - tau*(0.94)) / 2, in points of price: S protected 0.8; S free 2.6.

**Each judged cell's own optimum at rho 0.95 (dev price; the least a rule made for that cell alone could pay), and the S floor (the price of S alone):**

| judged cell | own optimum, S free | own optimum, S protected | S floor (dev) | S floor (eval) | S-alone reach (dev) |
|---|---|---|---|---|---|
| multihoprag/RK/GP | 34.6 | 34.6 | 0.0 | 0.0 | 0.0 |
| musique/RK/GP | 71.5 | 71.5 | 0.0 | 0.0 | 0.0 |
| musique/RK/CH | 71.1 | 71.1 | 0.0 | 0.0 | 0.0 |
| metaqa/SRK/GP | 17.4 | 83.1 | 71.6 | 72.3 | 4.1 |
| webqsp/SRK/GP | 25.2 | 52.2 | 36.9 | 38.5 | 36.2 |
| hotpotqa/RK/GP | 64.7 | 64.7 | 0.0 | 0.0 | 0.0 |
| hotpotqa/SRK/GP | 43.9 | 53.8 | 53.8 | 52.1 | 99.7 |
| 2wiki/RK/GP | 42.6 | 42.6 | 0.0 | 0.0 | 0.0 |
| 2wiki/SRK/GP | 37.8 | 40.3 | 40.0 | 35.3 | 93.8 |

**The cost in reach of a price (max over x of the worst judged cell's dev reach, at price <= tau for every cell):**

| price cap tau | S protected | S free |
|---|---|---|
| 0.30 | infeasible | 58.8 |
| 0.40 | infeasible | 67.8 |
| 0.50 | infeasible | 76.1 |
| 0.60 | infeasible | 83.6 |
| 0.80 | 72.3 | 92.7 |

**The integer rule, S protected (the call), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 97.5; kept 102 of 115 keys; dropped: K.knn|0, K.knn|2, R.mention+K.knn|2, R.mention+K.knn|3, R.mention|0, R.name_comention|0, R.title_comention+K.knn|0, R.title_comention+K.knn|1, R.title_comention+K.knn|2, R.title_comention+K.knn|3, R.title_comention+K.knn|4, R.title_comention|0, R.title_comention|1).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 83.6 | 99.9 / 84.4 | OUT |
| musique/RK/GP | 95.5 / 82.1 | 96.5 / 86.0 | OUT |
| musique/RK/CH | 95.4 / 82.1 | 96.5 / 86.0 | OUT |
| metaqa/SRK/GP | 95.2 / 97.5 | 95.5 / 97.6 | OUT |
| webqsp/SRK/GP | 98.8 / 89.1 | 98.7 / 90.5 | OUT |
| hotpotqa/RK/GP | 96.1 / 80.0 | 95.8 / 63.2 | OUT |
| hotpotqa/SRK/GP | 99.9 / 90.2 | 100.0 / 81.2 | OUT |
| 2wiki/RK/GP | 98.5 / 81.3 | 98.7 / 80.6 | OUT |
| 2wiki/SRK/GP | 99.9 / 87.4 | 99.8 / 86.9 | OUT |

**The integer rule, S free (beside), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 92.4; kept 98 of 115 keys; dropped: K.knn|0, K.knn|2, R.mention+K.knn|2, R.mention+K.knn|3, R.mention|0, R.name_comention|0, R.title_comention+K.knn|2, R.title_comention+K.knn|3, R.title_comention+K.knn|4, R.title_comention|0, S+R.mention+K.knn|2, S+R.mention+K.knn|3, S+R.mention+K.knn|4, S+R.mention|0, S+R.title_comention+K.knn|2, S+R.title_comention+K.knn|3, S+R.title_comention+K.knn|4).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 83.6 | 99.9 / 84.5 | OUT |
| musique/RK/GP | 95.5 / 82.2 | 96.8 / 86.1 | OUT |
| musique/RK/CH | 95.4 / 82.2 | 96.8 / 86.1 | OUT |
| metaqa/SRK/GP | 95.2 / 92.4 | 95.5 / 92.6 | OUT |
| webqsp/SRK/GP | 97.4 / 78.2 | 96.3 / 79.0 | OUT |
| hotpotqa/RK/GP | 96.1 / 80.1 | 95.9 / 63.2 | OUT |
| hotpotqa/SRK/GP | 95.5 / 82.2 | 95.1 / 69.5 | OUT |
| 2wiki/RK/GP | 98.7 / 81.4 | 98.7 / 80.7 | OUT |
| 2wiki/SRK/GP | 95.3 / 81.7 | 95.8 / 82.2 | OUT |

## Layer Q: the depth allocation (dev; an oracle bound, never a method)

Mean nodes per question at ALL >= 0.95 on the dev carve, by rule. **Uniform** = one level for every question (the family's own table); **hull** = a randomised uniform rule; **oracle depth** = each question gets the cheapest level that holds all its golds (the LP optimum; it reads the golds).

| dataset | family cells | uniform (least nodes at 0.95) | hull | oracle depth (family) | oracle ALL at 2,500 | uniform ALL at 2,500 | family ceiling on ALL | verdict at 2,500 |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 8 | 50 | 50 | 50 | 99.6 | 99.6 | 99.6 | FEASIBLE_AT_2500 |
| squad | 1 | 15 | 15 | 15 | 95.2 | 95.2 | 95.2 | FEASIBLE_AT_2500 |
| hotpotqa | 7 | 77 | 65 | 64 | 96.8 | 96.8 | 96.8 | FEASIBLE_AT_2500 |
| 2wiki | 7 | 908 | 899 | 561 | 95.5 | 95.5 | 95.5 | FEASIBLE_AT_2500 |
| musique | 10 | 15,400 | 14,660 | 3,382 | 78.4 | 78.4 | 96.3 | INFEASIBLE_AT_2500 |
| metaqa | 10 | 4,984 | 4,921 | 2,804 | 88.9 | 46.4 | 96.1 | INFEASIBLE_AT_2500 |
| webqsp | 12 | 39,572 | 38,940 | 10,250 | 55.5 | 55.5 | 95.6 | INFEASIBLE_AT_2500 |

**The menu each program had (the family's levels, its smallest level's mean nodes, and whether the nesting held):**

| dataset | levels | smallest level (mean nodes) | questions where a larger cut held less | G4-4 cells checked |
|---|---|---|---|---|
| multihoprag | 8 | 50 | 0 | 8 |
| squad | 1 | 15 | 0 | 1 |
| hotpotqa | 7 | 64 | 0 | 7 |
| 2wiki | 7 | 450 | 0 | 7 |
| musique | 10 | 2,500 | 0 | 4 |
| metaqa | 10 | 50 | 0 | 8 |
| webqsp | 12 | 2,500 | 0 | 4 |

**ALL (percent) at a mean-node cap, oracle depth in the family (uniform in brackets):**

| dataset | 2,500 | 3,500 | 5,000 | 7,000 | 10,000 |
|---|---|---|---|---|---|
| multihoprag | 99.6 (99.6) | 99.6 (99.6) | 99.6 (99.6) | 99.6 (99.6) | 99.6 (99.6) |
| squad | 95.2 (95.2) | 95.2 (95.2) | 95.2 (95.2) | 95.2 (95.2) | 95.2 (95.2) |
| hotpotqa | 96.8 (96.8) | 96.8 (96.8) | 96.8 (96.8) | 96.8 (96.8) | 96.8 (96.8) |
| 2wiki | 95.5 (95.5) | 95.5 (95.5) | 95.5 (95.5) | 95.5 (95.5) | 95.5 (95.5) |
| musique | 78.4 (78.4) | 95.8 (81.9) | 96.3 (84.5) | 96.3 (88.4) | 96.3 (91.7) |
| metaqa | 88.9 (46.4) | 96.1 (63.1) | 96.1 (96.1) | 96.1 (96.1) | 96.1 (96.1) |
| webqsp | 55.5 (55.5) | 73.3 (58.7) | 80.5 (62.2) | 86.4 (65.9) | 94.3 (69.5) |

**Share of questions whose cheapest covering pool in the family is at most N nodes (percent):**

| dataset | 2,500 | 3,500 | 5,000 | 7,000 | 10,000 | covered by no level |
|---|---|---|---|---|---|---|
| multihoprag | 99.6 | 99.6 | 99.6 | 99.6 | 99.6 | 0.4 |
| squad | 95.2 | 95.2 | 95.2 | 95.2 | 95.2 | 4.8 |
| hotpotqa | 96.8 | 96.8 | 96.8 | 96.8 | 96.8 | 3.2 |
| 2wiki | 95.5 | 95.5 | 95.5 | 95.5 | 95.5 | 4.5 |
| musique | 78.4 | 81.9 | 84.5 | 88.4 | 91.7 | 3.7 |
| metaqa | 46.4 | 63.1 | 63.6 | 96.1 | 96.1 | 3.9 |
| webqsp | 55.5 | 58.7 | 62.2 | 65.9 | 69.5 | 4.4 |

**The price of ALL, and the optimistic bound beside.** Nodes per point of ALL at 0.95 (oracle depth, family) and the optimistic all-cell bound (a question may pick a lucky small cell of ~6,000):

| dataset | nodes per point of ALL | oracle depth at 0.98 (family) | all-cell oracle at 0.95 (optimistic) | all-cell ALL at 2,500 (optimistic) |
|---|---|---|---|---|
| multihoprag | 0 | 50 | 11 | 100.0 |
| squad | - | - | 5 | 100.0 |
| hotpotqa | 0 | - | 20 | 99.9 |
| 2wiki | - | - | 42 | 98.6 |
| musique | 137 | - | 133 | 98.7 |
| metaqa | 50 | - | 485 | 98.7 |
| webqsp | - | - | 185 | 97.0 |

