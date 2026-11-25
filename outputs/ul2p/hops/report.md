# UL2H: the hop-coverage ledger, report

Declared in `docs/UL2H_HOP_COVERAGE_LEDGER.md`. Read: multihoprag, squad. Points with 95% intervals over questions, in percent; nodes are the mean uncapped pool size (hop 3: `lb` = unread questions filled by the 2-hop size; `*` = some question unread).

## Expectations

| id | call | detail |
|---|---|---|
| H1 | HELD | multihoprag: ALL in any at 3 hops 1.0; squad: ALL in any at 3 hops 0.99874 |
| H2 | HELD | multihoprag: G95 ('label:R.name_comention', 1, 273.884); squad: G95 ('label:K.knn', 2, 77.934) |
| H3 | NOT_HELD | multihoprag: cheapest label at hop 1 label:R.mention (0.0055); squad: cheapest label at hop 1 label:R.title_comention (0.1714) |
| H4 | HELD | squad: 15 beyond 3 hops in any; all reaches 13 |
| H5 | HELD | multihoprag: ALL in any at 1 hop 0.98584 |

## squad (RK, 11873 questions, 11873 golds, 973 missed by the seeds, mean seeds 7.5134)

Seeds only: ALL 91.81 [91.29, 92.28], ANY 91.81 [91.29, 92.28], recall 91.81 [91.29, 92.28].

Graph floors: G95 {'unit': 'label:K.knn', 'hop': 2, 'ALL': 0.95191, 'nodes': 77.934, 'bound': False}; G98 {'unit': 'any', 'hop': 2, 'ALL': 0.98796, 'nodes': 6511.628, 'bound': False}; best ALL {'unit': 'all', 'hop': 3, 'ALL': 0.99983, 'nodes': 19813.719, 'bound': False}.

| unit | pairs | hop | ALL | ANY | recall | nodes | price/seed |
|---|---:|---:|---|---|---|---:|---:|
| label:R.mention | 878,200 | 1 | 95.44 [95.06, 95.81] | 95.44 [95.06, 95.81] | 95.44 [95.06, 95.81] | 343.988 | 44.783 |
| label:R.mention | 878,200 | 2 | 97.08 [96.77, 97.37] | 97.08 [96.77, 97.37] | 97.08 [96.77, 97.37] | 3336.295 |  |
| label:R.mention | 878,200 | 3 | 98.75 [98.54, 98.95] | 98.75 [98.54, 98.95] | 98.75 [98.54, 98.95] | 11053.434 |  |
| label:R.name_comention | 173,645 | 1 | 94.24 [93.82, 94.65] | 94.24 [93.82, 94.65] | 94.24 [93.82, 94.65] | 108.606 | 13.4549 |
| label:R.name_comention | 173,645 | 2 | 95.81 [95.46, 96.15] | 95.81 [95.46, 96.15] | 95.81 [95.46, 96.15] | 1475.914 |  |
| label:R.name_comention | 173,645 | 3 | 97.61 [97.34, 97.88] | 97.61 [97.34, 97.88] | 97.61 [97.34, 97.88] | 8151.74 |  |
| label:R.title_comention | 2,809 | 1 | 91.81 [91.29, 92.28] | 91.81 [91.29, 92.28] | 91.81 [91.29, 92.28] | 8.801 | 0.1714 |
| label:R.title_comention | 2,809 | 2 | 91.83 [91.32, 92.29] | 91.83 [91.32, 92.29] | 91.83 [91.32, 92.29] | 9.851 |  |
| label:R.title_comention | 2,809 | 3 | 91.83 [91.32, 92.29] | 91.83 [91.32, 92.29] | 91.83 [91.32, 92.29] | 9.852 |  |
| label:K.knn | 43,350 | 1 | 94.24 [93.81, 94.65] | 94.24 [93.81, 94.65] | 94.24 [93.81, 94.65] | 30.246 | 3.0256 |
| label:K.knn | 43,350 | 2 | 95.19 [94.78, 95.59] | 95.19 [94.78, 95.59] | 95.19 [94.78, 95.59] | 77.934 |  |
| label:K.knn | 43,350 | 3 | 95.92 [95.56, 96.29] | 95.92 [95.56, 96.29] | 95.92 [95.56, 96.29] | 171.323 |  |
| any | 1,098,004 | 1 | 96.40 [96.06, 96.75] | 96.40 [96.06, 96.75] | 96.40 [96.06, 96.75] | 413.428 | 54.0252 |
| any | 1,098,004 | 2 | 98.80 [98.59, 98.99] | 98.80 [98.59, 98.99] | 98.80 [98.59, 98.99] | 6511.628 |  |
| any | 1,098,004 | 3 | 99.87 [99.81, 99.93] | 99.87 [99.81, 99.93] | 99.87 [99.81, 99.93] | 17963.409 |  |
| plus:H5 | 1,158,685 | 1 | 96.83 [96.51, 97.15] | 96.83 [96.51, 97.15] | 96.83 [96.51, 97.15] | 441.934 | 57.8192 |
| plus:H5 | 1,158,685 | 2 | 99.20 [99.04, 99.35] | 99.20 [99.04, 99.35] | 99.20 [99.04, 99.35] | 7418.394 |  |
| plus:H5 | 1,158,685 | 3 | 99.95 [99.91, 99.98] | 99.95 [99.91, 99.98] | 99.95 [99.91, 99.98] | 18735.365 |  |
| plus:HM | 1,132,191 | 1 | 96.72 [96.39, 97.04] | 96.72 [96.39, 97.04] | 96.72 [96.39, 97.04] | 429.214 | 56.1262 |
| plus:HM | 1,132,191 | 2 | 99.07 [98.90, 99.24] | 99.07 [98.90, 99.24] | 99.07 [98.90, 99.24] | 6997.561 |  |
| plus:HM | 1,132,191 | 3 | 99.92 [99.87, 99.97] | 99.92 [99.87, 99.97] | 99.92 [99.87, 99.97] | 18355.381 |  |
| plus:HK1 | 1,136,717 | 1 | 96.70 [96.38, 97.02] | 96.70 [96.38, 97.02] | 96.70 [96.38, 97.02] | 432.338 | 56.542 |
| plus:HK1 | 1,136,717 | 2 | 99.07 [98.89, 99.23] | 99.07 [98.89, 99.23] | 99.07 [98.89, 99.23] | 7123.325 |  |
| plus:HK1 | 1,136,717 | 3 | 99.93 [99.88, 99.98] | 99.93 [99.88, 99.98] | 99.93 [99.88, 99.98] | 18407.885 |  |
| plus:NTc | 1,170,857 | 1 | 96.61 [96.29, 96.93] | 96.61 [96.29, 96.93] | 96.61 [96.29, 96.93] | 457.437 | 59.8826 |
| plus:NTc | 1,170,857 | 2 | 99.11 [98.94, 99.27] | 99.11 [98.94, 99.27] | 99.11 [98.94, 99.27] | 8327.183 |  |
| plus:NTc | 1,170,857 | 3 | 99.95 [99.91, 99.98] | 99.95 [99.91, 99.98] | 99.95 [99.91, 99.98] | 19226.987 |  |
| plus:NT1 | 1,153,065 | 1 | 96.70 [96.38, 97.02] | 96.70 [96.38, 97.02] | 96.70 [96.38, 97.02] | 442.916 | 57.9498 |
| plus:NT1 | 1,153,065 | 2 | 99.12 [98.95, 99.28] | 99.12 [98.95, 99.28] | 99.12 [98.95, 99.28] | 7580.583 |  |
| plus:NT1 | 1,153,065 | 3 | 99.97 [99.93, 99.99] | 99.97 [99.93, 99.99] | 99.97 [99.93, 99.99] | 18624.728 |  |
| plus:HK2 | 1,170,733 | 1 | 96.80 [96.47, 97.12] | 96.80 [96.47, 97.12] | 96.80 [96.47, 97.12] | 449.275 | 58.7963 |
| plus:HK2 | 1,170,733 | 2 | 99.14 [98.96, 99.29] | 99.14 [98.96, 99.29] | 99.14 [98.96, 99.29] | 7605.773 |  |
| plus:HK2 | 1,170,733 | 3 | 99.94 [99.89, 99.98] | 99.94 [99.89, 99.98] | 99.94 [99.89, 99.98] | 18668.881 |  |
| plus:H10 | 1,214,932 | 1 | 96.98 [96.66, 97.28] | 96.98 [96.66, 97.28] | 96.98 [96.66, 97.28] | 469.034 | 61.426 |
| plus:H10 | 1,214,932 | 2 | 99.33 [99.17, 99.46] | 99.33 [99.17, 99.46] | 99.33 [99.17, 99.46] | 8206.403 |  |
| plus:H10 | 1,214,932 | 3 | 99.96 [99.92, 99.99] | 99.96 [99.92, 99.99] | 99.96 [99.92, 99.99] | 19141.394 |  |
| all | 1,549,156 | 1 | 97.30 [97.00, 97.59] | 97.30 [97.00, 97.59] | 97.30 [97.00, 97.59] | 544.531 | 71.4744 |
| all | 1,549,156 | 2 | 99.51 [99.39, 99.63] | 99.51 [99.39, 99.63] | 99.51 [99.39, 99.63] | 10487.947 |  |
| all | 1,549,156 | 3 | 99.98 [99.96, 100.00] | 99.98 [99.96, 100.00] | 99.98 [99.96, 100.00] | 19813.719 |  |

Missed golds by distance in the any-link graph: 1: 546 (56.12 [52.83, 59.30]); 2: 284 (29.19 [26.41, 32.07]); 3: 128 (13.15 [11.00, 15.31]); 4: 15 (1.54 [0.82, 2.36]); isolated in all: 0.

| hop | reached by any | unreached | rescue all | ALL_ONLY | NONE | ISOLATED | FAR | MULTI_ALONE | MIXED |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 546 | 427 | 106 | 0 | 321 | 0 | 321 | 340 | 0 |
| 2 | 830 | 143 | 85 | 1 | 58 | 0 | 58 | 520 | 79 |
| 3 | 958 | 15 | 13 | 0 | 2 | 0 | 2 | 753 | 51 |

| hop | form | rescued golds | share | nodes vs any | low end above 0 |
|---:|---|---:|---|---:|---|
| 1 | H5 | 51 | 5.24 [3.80, 6.68] | 28.506 | True |
| 1 | HM | 38 | 3.91 [2.67, 5.14] | 15.786 | True |
| 1 | HK1 | 35 | 3.60 [2.47, 4.73] | 18.91 | True |
| 1 | NTc | 24 | 2.47 [1.54, 3.49] | 44.009 | True |
| 1 | NT1 | 35 | 3.60 [2.47, 4.73] | 29.488 | True |
| 1 | HK2 | 47 | 4.83 [3.60, 6.27] | 35.847 | True |
| 1 | H10 | 68 | 6.99 [5.34, 8.63] | 55.606 | True |
| 2 | H5 | 48 | 4.93 [3.60, 6.37] | 906.765 | True |
| 2 | HM | 33 | 3.39 [2.26, 4.62] | 485.933 | True |
| 2 | HK1 | 33 | 3.39 [2.26, 4.73] | 611.697 | True |
| 2 | NTc | 37 | 3.80 [2.67, 5.04] | 1815.555 | True |
| 2 | NT1 | 38 | 3.91 [2.77, 5.14] | 1068.955 | True |
| 2 | HK2 | 41 | 4.21 [2.98, 5.55] | 1094.144 | True |
| 2 | H10 | 63 | 6.48 [4.93, 8.12] | 1694.775 | True |
| 3 | H5 | 9 | 0.92 [0.41, 1.54] | 771.956 | True |
| 3 | HM | 6 | 0.62 [0.21, 1.13] | 391.971 | True |
| 3 | HK1 | 7 | 0.72 [0.31, 1.34] | 444.476 | True |
| 3 | NTc | 9 | 0.92 [0.41, 1.54] | 1263.578 | True |
| 3 | NT1 | 11 | 1.13 [0.51, 1.85] | 661.318 | True |
| 3 | HK2 | 8 | 0.82 [0.31, 1.44] | 705.472 | True |
| 3 | H10 | 10 | 1.03 [0.51, 1.75] | 1177.984 | True |

Candidates named (not adopted): ADD form H5 at 1 hop(s): rescues 51 missed golds (5.24%) for 28.506 more nodes per question; ADD form HM at 1 hop(s): rescues 38 missed golds (3.91%) for 15.786 more nodes per question; ADD form HK1 at 1 hop(s): rescues 35 missed golds (3.60%) for 18.91 more nodes per question; ADD form NTc at 1 hop(s): rescues 24 missed golds (2.47%) for 44.009 more nodes per question; ADD form NT1 at 1 hop(s): rescues 35 missed golds (3.60%) for 29.488 more nodes per question; ADD form HK2 at 1 hop(s): rescues 47 missed golds (4.83%) for 35.847 more nodes per question; ADD form H10 at 1 hop(s): rescues 68 missed golds (6.99%) for 55.606 more nodes per question; PRIORITISE label label:R.mention at 1 hop(s): it alone links the most golds the other labels alone do not; ADD form H5 at 2 hop(s): rescues 48 missed golds (4.93%) for 906.765 more nodes per question; ADD form HM at 2 hop(s): rescues 33 missed golds (3.39%) for 485.933 more nodes per question; ADD form HK1 at 2 hop(s): rescues 33 missed golds (3.39%) for 611.697 more nodes per question; ADD form NTc at 2 hop(s): rescues 37 missed golds (3.80%) for 1815.555 more nodes per question; ADD form NT1 at 2 hop(s): rescues 38 missed golds (3.91%) for 1068.955 more nodes per question; ADD form HK2 at 2 hop(s): rescues 41 missed golds (4.21%) for 1094.144 more nodes per question; ADD form H10 at 2 hop(s): rescues 63 missed golds (6.48%) for 1694.775 more nodes per question; PRIORITISE label label:R.mention at 2 hop(s): it alone links the most golds the other labels alone do not; ADD form H5 at 3 hop(s): rescues 9 missed golds (0.92%) for 771.956 more nodes per question; ADD form HM at 3 hop(s): rescues 6 missed golds (0.62%) for 391.971 more nodes per question; ADD form HK1 at 3 hop(s): rescues 7 missed golds (0.72%) for 444.476 more nodes per question; ADD form NTc at 3 hop(s): rescues 9 missed golds (0.92%) for 1263.578 more nodes per question; ADD form NT1 at 3 hop(s): rescues 11 missed golds (1.13%) for 661.318 more nodes per question; ADD form HK2 at 3 hop(s): rescues 8 missed golds (0.82%) for 705.472 more nodes per question; ADD form H10 at 3 hop(s): rescues 10 missed golds (1.03%) for 1177.984 more nodes per question; PRIORITISE label label:R.mention at 3 hop(s): it alone links the most golds the other labels alone do not

## multihoprag (RK, 1130 questions, 2985 golds, 851 missed by the seeds, mean seeds 7.4398)

Seeds only: ALL 44.51 [41.59, 47.43], ANY 97.97 [97.08, 98.76], recall 71.49 [70.00, 73.03].

Graph floors: G95 {'unit': 'label:R.name_comention', 'hop': 1, 'ALL': 0.98584, 'nodes': 273.884, 'bound': False}; G98 {'unit': 'label:R.name_comention', 'hop': 1, 'ALL': 0.98584, 'nodes': 273.884, 'bound': False}; best ALL {'unit': 'label:R.name_comention', 'hop': 2, 'ALL': 1.0, 'nodes': 607.816, 'bound': False}.

| unit | pairs | hop | ALL | ANY | recall | nodes | price/seed |
|---|---:|---:|---|---|---|---:|---:|
| label:R.mention | 2 | 1 | 44.60 [41.68, 47.52] | 97.97 [97.08, 98.76] | 71.52 [70.04, 73.08] | 7.481 | 0.0055 |
| label:R.mention | 2 | 2 | 44.60 [41.68, 47.52] | 97.97 [97.08, 98.76] | 71.52 [70.04, 73.08] | 7.481 |  |
| label:R.mention | 2 | 3 | 44.60 [41.68, 47.52] | 97.97 [97.08, 98.76] | 71.52 [70.04, 73.08] | 7.481 |  |
| label:R.name_comention | 25,005 | 1 | 98.58 [97.88, 99.20] | 100.00 [100.00, 100.00] | 99.46 [99.19, 99.70] | 273.884 | 35.8133 |
| label:R.name_comention | 25,005 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 607.816 |  |
| label:R.name_comention | 25,005 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.002 |  |
| label:R.title_comention | 75 | 1 | 45.40 [42.48, 48.32] | 97.97 [97.08, 98.76] | 71.86 [70.33, 73.39] | 8.913 | 0.198 |
| label:R.title_comention | 75 | 2 | 47.79 [44.78, 50.71] | 98.14 [97.34, 98.85] | 73.27 [71.74, 74.80] | 22.649 |  |
| label:R.title_comention | 75 | 3 | 48.58 [45.57, 51.50] | 98.14 [97.34, 98.85] | 73.60 [72.10, 75.13] | 25.271 |  |
| label:K.knn | 1,366 | 1 | 65.22 [62.48, 68.14] | 99.82 [99.56, 100.00] | 86.16 [85.13, 87.26] | 24.726 | 2.3234 |
| label:K.knn | 1,366 | 2 | 72.21 [69.65, 74.87] | 99.82 [99.56, 100.00] | 89.38 [88.48, 90.35] | 48.959 |  |
| label:K.knn | 1,366 | 3 | 75.22 [72.74, 77.79] | 99.82 [99.56, 100.00] | 90.52 [89.65, 91.39] | 90.19 |  |
| any | 26,448 | 1 | 98.58 [97.88, 99.20] | 100.00 [100.00, 100.00] | 99.46 [99.19, 99.70] | 275.576 | 36.0407 |
| any | 26,448 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.694 |  |
| any | 26,448 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:H5 | 28,835 | 1 | 98.94 [98.32, 99.47] | 100.00 [100.00, 100.00] | 99.60 [99.36, 99.80] | 289.196 | 37.8714 |
| plus:H5 | 28,835 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.749 |  |
| plus:H5 | 28,835 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:HM | 27,780 | 1 | 98.85 [98.14, 99.47] | 100.00 [100.00, 100.00] | 99.56 [99.30, 99.80] | 281.57 | 36.8463 |
| plus:HM | 27,780 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.711 |  |
| plus:HM | 27,780 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:HK1 | 29,160 | 1 | 98.94 [98.32, 99.47] | 100.00 [100.00, 100.00] | 99.60 [99.36, 99.80] | 292.578 | 38.3259 |
| plus:HK1 | 29,160 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.758 |  |
| plus:HK1 | 29,160 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:NTc | 40,116 | 1 | 99.20 [98.67, 99.65] | 100.00 [100.00, 100.00] | 99.70 [99.50, 99.87] | 376.535 | 49.6107 |
| plus:NTc | 40,116 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:NTc | 40,116 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:NT1 | 31,052 | 1 | 99.29 [98.76, 99.73] | 100.00 [100.00, 100.00] | 99.73 [99.53, 99.90] | 312.204 | 40.964 |
| plus:NT1 | 31,052 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.794 |  |
| plus:NT1 | 31,052 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:HK2 | 31,334 | 1 | 99.20 [98.58, 99.65] | 100.00 [100.00, 100.00] | 99.70 [99.47, 99.87] | 308.13 | 40.4163 |
| plus:HK2 | 31,334 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.778 |  |
| plus:HK2 | 31,334 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| plus:H10 | 31,036 | 1 | 99.11 [98.50, 99.65] | 100.00 [100.00, 100.00] | 99.67 [99.43, 99.87] | 303.258 | 39.7614 |
| plus:H10 | 31,036 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.778 |  |
| plus:H10 | 31,036 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| all | 60,625 | 1 | 99.82 [99.56, 100.00] | 100.00 [100.00, 100.00] | 99.93 [99.83, 100.00] | 415.142 | 54.7999 |
| all | 60,625 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |
| all | 60,625 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |  |

Missed golds by distance in the any-link graph: 1: 835 (98.12 [97.18, 98.95]); 2: 16 (1.88 [1.05, 2.82]); 3: 0 (0.00 [0.00, 0.00]); 4: 0 (0.00 [0.00, 0.00]); isolated in all: 0.

| hop | reached by any | unreached | rescue all | ALL_ONLY | NONE | ISOLATED | FAR | MULTI_ALONE | MIXED |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 835 | 16 | 14 | 0 | 2 | 0 | 2 | 444 | 0 |
| 2 | 851 | 0 | 0 | 0 | 0 | 0 | 0 | 538 | 0 |
| 3 | 851 | 0 | 0 | 0 | 0 | 0 | 0 | 573 | 0 |

| hop | form | rescued golds | share | nodes vs any | low end above 0 |
|---:|---|---:|---|---:|---|
| 1 | H5 | 4 | 0.47 [0.12, 0.95] | 13.62 | True |
| 1 | HM | 3 | 0.35 [0.00, 0.81] | 5.994 | False |
| 1 | HK1 | 4 | 0.47 [0.12, 0.95] | 17.002 | True |
| 1 | NTc | 7 | 0.82 [0.24, 1.44] | 100.958 | True |
| 1 | NT1 | 8 | 0.94 [0.35, 1.64] | 36.628 | True |
| 1 | HK2 | 7 | 0.82 [0.34, 1.51] | 32.554 | True |
| 1 | H10 | 6 | 0.70 [0.23, 1.31] | 27.681 | True |
| 2 | H5 | 0 | 0.00 [0.00, 0.00] | 0.055 | False |
| 2 | HM | 0 | 0.00 [0.00, 0.00] | 0.017 | False |
| 2 | HK1 | 0 | 0.00 [0.00, 0.00] | 0.064 | False |
| 2 | NTc | 0 | 0.00 [0.00, 0.00] | 0.306 | False |
| 2 | NT1 | 0 | 0.00 [0.00, 0.00] | 0.1 | False |
| 2 | HK2 | 0 | 0.00 [0.00, 0.00] | 0.084 | False |
| 2 | H10 | 0 | 0.00 [0.00, 0.00] | 0.084 | False |
| 3 | H5 | 0 | 0.00 [0.00, 0.00] | 0.0 | False |
| 3 | HM | 0 | 0.00 [0.00, 0.00] | 0.0 | False |
| 3 | HK1 | 0 | 0.00 [0.00, 0.00] | 0.0 | False |
| 3 | NTc | 0 | 0.00 [0.00, 0.00] | 0.0 | False |
| 3 | NT1 | 0 | 0.00 [0.00, 0.00] | 0.0 | False |
| 3 | HK2 | 0 | 0.00 [0.00, 0.00] | 0.0 | False |
| 3 | H10 | 0 | 0.00 [0.00, 0.00] | 0.0 | False |

Candidates named (not adopted): ADD form H5 at 1 hop(s): rescues 4 missed golds (0.47%) for 13.62 more nodes per question; ADD form HK1 at 1 hop(s): rescues 4 missed golds (0.47%) for 17.002 more nodes per question; ADD form NTc at 1 hop(s): rescues 7 missed golds (0.82%) for 100.958 more nodes per question; ADD form NT1 at 1 hop(s): rescues 8 missed golds (0.94%) for 36.628 more nodes per question; ADD form HK2 at 1 hop(s): rescues 7 missed golds (0.82%) for 32.554 more nodes per question; ADD form H10 at 1 hop(s): rescues 6 missed golds (0.70%) for 27.681 more nodes per question; PRIORITISE label label:R.name_comention at 1 hop(s): it alone links the most golds the other labels alone do not; PRIORITISE label label:R.name_comention at 2 hop(s): it alone links the most golds the other labels alone do not; PRIORITISE label label:R.name_comention at 3 hop(s): it alone links the most golds the other labels alone do not
