# AP3 Part A: the ladder (declared in docs/AP3_STRUCTURAL_SLOT_TITLE_FORMS.md)

Graph level, no model. R = b1w's structural family (the filed W2_S pairs). A rung adds to R the base-title pairs (U2b's rule, ambiguous surfaces dropped) whose surface occurs in at most `cap` passages of the setting. Gain = slots newly linked by the structural family, in points of the setting's slots (GP: gold pairs; MuSiQue: chain slots); price = pairs added per slot added. S1 carries the deepest rung up to which every rung's price is at most R's own pairs per slot linked (p0) and whose gain is at least 2.0 points.

## 2wiki (GP, 2175 slots, 6119 passages)

R: 2030 pairs link 872 slots (40.09%); p0 = 2.328 pairs per slot; S precision of R 0.8246, S recall 0.8438. **S1: None** (the deepest rung that passes the price adds 13 slots, below 2.0 points of the setting's 2175).

| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 30 | 13 | +0.60 [+0.28, +0.94] ABOVE | 40.69 | 2.31 | 15/30 | 2 | 0 |
| 2 | 191 | 50 | +2.30 [+1.69, +2.96] ABOVE | 42.39 | 3.82 | 72/191 | 25 | 0 |
| 3 | 300 | 64 | +2.94 [+2.25, +3.68] ABOVE | 43.03 | 4.69 | 95/300 | 55 | 0 |
| 4 | 396 | 69 | +3.17 [+2.45, +3.93] ABOVE | 43.26 | 5.74 | 111/396 | 90 | 0 |
| 6 | 533 | 73 | +3.36 [+2.61, +4.12] ABOVE | 43.45 | 7.3 | 129/533 | 128 | 1 |
| 8 | 679 | 76 | +3.49 [+2.75, +4.29] ABOVE | 43.59 | 8.93 | 134/679 | 201 | 1 |
| 12 | 786 | 80 | +3.68 [+2.92, +4.49] ABOVE | 43.77 | 9.82 | 142/786 | 253 | 1 |
| 16 | 829 | 80 | +3.68 [+2.92, +4.49] ABOVE | 43.77 | 10.36 | 142/829 | 279 | 1 |
| 25 | 1005 | 81 | +3.72 [+2.97, +4.55] ABOVE | 43.82 | 12.41 | 145/1005 | 385 | 1 |
| inf | 2281 | 82 | +3.77 [+3.00, +4.60] ABOVE | 43.86 | 27.82 | 148/2281 | 1503 | 1 |

## hotpotqa (GP, 1000 slots, 9811 passages)

R: 4315 pairs link 591 slots (59.1%); p0 = 7.301 pairs per slot; S precision of R 0.8197, S recall 0.8039. **S1: 3** (carried).

| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 15 | 5 | +0.50 [+0.10, +1.00] ABOVE | 59.6 | 3.0 | 7/15 | 4 | 2 |
| 2 | 215 | 43 | +4.30 [+3.10, +5.50] ABOVE | 63.4 | 5.0 | 90/215 | 45 | 3 |
| 3 | 357 | 54 | +5.40 [+4.00, +6.70] ABOVE | 64.5 | 6.61 | 129/357 | 79 | 3 |
| 4 | 495 | 57 | +5.70 [+4.30, +7.00] ABOVE | 64.8 | 8.68 | 158/495 | 122 | 3 |
| 6 | 683 | 61 | +6.10 [+4.70, +7.50] ABOVE | 65.2 | 11.2 | 189/683 | 202 | 3 |
| 8 | 895 | 64 | +6.40 [+4.90, +7.80] ABOVE | 65.5 | 13.98 | 213/895 | 297 | 3 |
| 12 | 1311 | 69 | +6.90 [+5.40, +8.40] ABOVE | 66.0 | 19.0 | 263/1311 | 449 | 3 |
| 16 | 1510 | 73 | +7.30 [+5.70, +8.80] ABOVE | 66.4 | 20.68 | 269/1510 | 582 | 3 |
| 25 | 1707 | 75 | +7.50 [+5.90, +9.10] ABOVE | 66.6 | 22.76 | 274/1707 | 715 | 3 |
| inf | 5244 | 77 | +7.70 [+6.10, +9.30] ABOVE | 66.8 | 68.1 | 283/5244 | 3937 | 4 |

## musique (CH, 1648 slots, 11656 passages)

R: 41429 pairs link 475 slots (28.82%); p0 = 87.219 pairs per slot; S precision of R 0.9234, S recall 0.8933. **S1: 25** (carried).

| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 29 | 1 | +0.06 [+0.00, +0.18] LEVEL | 28.88 | 29.0 | 0/29 | 11 | 0 |
| 2 | 154 | 16 | +0.97 [+0.54, +1.45] ABOVE | 29.79 | 9.62 | 0/154 | 43 | 0 |
| 3 | 272 | 24 | +1.46 [+0.91, +2.05] ABOVE | 30.28 | 11.33 | 1/272 | 88 | 0 |
| 4 | 348 | 33 | +2.00 [+1.38, +2.68] ABOVE | 30.83 | 10.55 | 1/348 | 124 | 0 |
| 6 | 545 | 35 | +2.12 [+1.48, +2.81] ABOVE | 30.95 | 15.57 | 1/545 | 211 | 0 |
| 8 | 682 | 37 | +2.25 [+1.58, +2.95] ABOVE | 31.07 | 18.43 | 1/682 | 271 | 0 |
| 12 | 951 | 45 | +2.73 [+2.02, +3.48] ABOVE | 31.55 | 21.13 | 1/951 | 418 | 0 |
| 16 | 1300 | 50 | +3.03 [+2.22, +3.94] ABOVE | 31.86 | 26.0 | 1/1300 | 618 | 0 |
| 25 | 1864 | 54 | +3.28 [+2.41, +4.18] ABOVE | 32.1 | 34.52 | 2/1864 | 897 | 0 |
| inf | 5844 | 58 | +3.52 [+2.62, +4.49] ABOVE | 32.34 | 100.76 | 2/5844 | 4576 | 0 |

Carried: 2wiki None, hotpotqa 3, musique 25.
