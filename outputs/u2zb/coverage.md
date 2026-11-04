# U2z stage B1: coverage of `rrf50+none/both/kt_cos:nbc_l2.8@0.9`

Written 2026-11-04T09:58:52Z. No model; the pools hold no gold. ALL and gold recall in percent with 95% intervals; rows per question; the corpus is the dataset's nodes (the setting's passages on B1), and HippoRAG 2's candidate set is every one of them. Pick O's pools (COV's) stand beside and decide nothing.

| dataset | carve | questions | ALL | gold recall | rows: mean / median / p90 | corpus | mean rows / corpus | pick O: ALL, recall, rows |
|---|---|---:|---|---|---|---:|---:|---|
| musique | s1eval | 2417 | 93.3 [92.3, 94.3] | 97.0 [96.6, 97.5] | 4021.0 / 4021 / 4021 | 117,534 | 3.421% | 94.8, 97.8, 5263.6 (this pick x0.76) |
| musique | b1w | 1000 | 97.7 [96.8, 98.6] | 99.1 [98.7, 99.4] | 1406.0 / 1406 / 1406 | 11,654 | 12.065% | 99.8, 99.9, 3433.6 (this pick x0.41) |
| musique | s1sel | 1534 | 96.2 [95.1, 97.1] | 98.4 [98.0, 98.8] | 4506.4 / 4526 / 4526 | 117,534 | 3.834% | none |
| hotpotqa | s1eval | 7405 | 98.2 [98.0, 98.5] | 99.1 [98.9, 99.2] | 2516.0 / 2516 / 2516 | 5,233,329 | 0.048% | 96.0, 97.9, 706.0 (this pick x3.56) |
| hotpotqa | b1w | 1000 | 99.9 [99.7, 100.0] | 100.0 [99.9, 100.0] | 1108.0 / 1108 / 1108 | 9,811 | 11.293% | 97.0, 98.5, 65.3 (this pick x16.97) |
| hotpotqa | s1sel | 1508 | 97.8 [97.0, 98.5] | 98.8 [98.4, 99.2] | 2533.0 / 2533 / 2533 | 5,233,329 | 0.048% | none |
| 2wiki | s1eval | 12576 | 98.6 [98.4, 98.8] | 99.4 [99.3, 99.5] | 3570.0 / 3570 / 3570 | 5,989,847 | 0.060% | 96.9, 98.7, 1081.2 (this pick x3.30) |
| 2wiki | b1w | 1000 | 99.8 [99.5, 100.0] | 99.9 [99.8, 100.0] | 1174.0 / 1174 / 1174 | 6,119 | 19.187% | 91.9, 96.1, 88.5 (this pick x13.27) |
| 2wiki | s1sel | 1496 | 98.6 [98.0, 99.1] | 99.4 [99.2, 99.6] | 3342.0 / 3342 / 3342 | 5,989,847 | 0.056% | none |

s1eval: the evaluation carve, the read that counts (the pick was not chosen on it); b1w: HippoRAG 2's 1,000-question setting of the dataset (its corpus is the setting's passages); s1sel: the carve the pick was chosen on, the reproduction of stage A (PKO built no pick O pools on it).
