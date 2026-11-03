# COV: pick O's coverage and localization record

Declared in docs/COV_PICK_O_COVERAGE_AND_LOCALIZATION.md. No model. 95% intervals over 2,000 question resamples (seed 0). Percent of questions (ALL) and of golds (gold recall).

## Pick O's pools

| carve | dataset | questions | ALL [95%] | gold recall [95%] | rows mean / median / p90 / max | corpus | mean rows / corpus | HippoRAG 2's candidates |
|---|---|---:|---|---|---|---:|---:|---:|
| s1eval | 2wiki | 12,576 | 96.9 [96.6, 97.2] | 98.7 [98.6, 98.8] | 1,081.2 / 1,090 / 1,383 / 1,853 | 5,989,847 | 0.0181% | 5,989,847 (all) |
| s1eval | hotpotqa | 7,405 | 96.0 [95.5, 96.4] | 97.9 [97.7, 98.1] | 706.0 / 688 / 1,017 / 1,637 | 5,233,329 | 0.0135% | 5,233,329 (all) |
| s1eval | musique | 2,417 | 94.8 [93.9, 95.6] | 97.8 [97.4, 98.1] | 5,263.6 / 4,980 / 6,490 / 12,333 | 117,534 | 4.4784% | 117,534 (all) |
| b1 | 2wiki | 1,000 | 91.9 [90.1, 93.6] | 96.1 [95.2, 96.9] | 88.5 / 77 / 145 / 261 | 6,119 | 1.4459% | 6,119 (all) |
| b1 | hotpotqa | 1,000 | 97.0 [95.9, 98.1] | 98.5 [98.0, 99.1] | 65.3 / 53 / 100 / 321 | 9,811 | 0.6651% | 9,811 (all) |
| b1 | musique | 1,000 | 99.8 [99.5, 100.0] | 99.9 [99.8, 100.0] | 3,433.6 / 3,455 / 3,532 / 3,707 | 11,654 | 29.4627% | 11,654 (all) |

## s1eval: other rankers at each pool's own size

Each order is read at depth min(k_q, its length), k_q = the question's pool rows. pool - order is paired by question; MORE / SAME / LESS by the interval.

| dataset | order | its gold recall | pool - it, recall [95%] | call | its ALL | pool - it, ALL [95%] | call | questions past its length |
|---|---|---:|---|---|---:|---|---|---:|
| 2wiki | A1 | 98.0 | +0.6 [+0.5, +0.8] | MORE | 95.4 | +1.5 [+1.1, +1.9] | MORE | 0 |
| 2wiki | A1_all | 98.1 | +0.6 [+0.4, +0.8] | MORE | 95.5 | +1.3 [+0.9, +1.7] | MORE | 0 |
| 2wiki | RRF | 76.4 | +22.3 [+21.9, +22.7] | MORE | 56.6 | +40.3 [+39.4, +41.1] | MORE | 5 |
| 2wiki | DENSE | 73.5 | +25.2 [+24.8, +25.6] | MORE | 52.8 | +44.1 [+43.2, +45.0] | MORE | 8,139 |
| 2wiki | SPLADE | 74.6 | +24.1 [+23.7, +24.5] | MORE | 52.8 | +44.0 [+43.1, +44.9] | MORE | 8,139 |
| hotpotqa | A1 | 99.5 | -1.6 [-1.8, -1.4] | LESS | 99.2 | -3.3 [-3.7, -2.9] | LESS | 0 |
| hotpotqa | A1_all | 99.4 | -1.5 [-1.7, -1.3] | LESS | 99.0 | -3.0 [-3.4, -2.6] | LESS | 0 |
| hotpotqa | RRF | 91.6 | +6.3 [+5.9, +6.7] | MORE | 83.3 | +12.6 [+11.9, +13.4] | MORE | 0 |
| hotpotqa | DENSE | 89.9 | +8.0 [+7.6, +8.4] | MORE | 80.3 | +15.7 [+14.9, +16.5] | MORE | 839 |
| hotpotqa | SPLADE | 88.2 | +9.7 [+9.2, +10.1] | MORE | 76.8 | +19.2 [+18.3, +20.1] | MORE | 839 |
| musique | A1 | 96.9 | +0.9 [+0.6, +1.2] | MORE | 93.1 | +1.7 [+1.0, +2.4] | MORE | 157 |
| musique | A1_all | 96.9 | +0.9 [+0.6, +1.2] | MORE | 93.1 | +1.7 [+1.0, +2.4] | MORE | 157 |
| musique | RRF | 88.6 | +9.2 [+8.4, +10.0] | MORE | 75.7 | +19.0 [+17.5, +20.6] | MORE | 2,417 |
| musique | DENSE | 85.7 | +12.0 [+11.2, +12.9] | MORE | 71.3 | +23.4 [+21.8, +25.2] | MORE | 2,417 |
| musique | SPLADE | 79.4 | +18.4 [+17.5, +19.3] | MORE | 56.7 | +38.1 [+36.2, +40.0] | MORE | 2,417 |

P3a's RRF top 10 inside the pool (the first stage agrees): 2wiki 100.00%, hotpotqa 100.00%, musique 100.00%.
