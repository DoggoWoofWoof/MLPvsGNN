# UL3 part A: the iteration subset, diagnostic report

Declared in `docs/UL3_LINK_TIGHTENING.md`. The keep-sets are chosen on the dev questions with index mod 4 = 0 and graded on the other dev questions; s1eval is not read. No call is filed from this report: it shows what each rule keeps and which classes it loses, so that the next declared rule targets the failures. Shares are points with 95% intervals over questions, in percent.

## multihoprag RK GP

any-link candidates / covered slots: subset 12,520 / 102; rest of dev 37,151 / 323.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 99.4 [98.5, 100.0] | 98.9 [98.7, 99.0] |
| P | 0.95 | 100.0 [100.0, 100.0] | 99.4 [98.5, 100.0] | 98.9 [98.7, 99.0] |
| P | 0.9 | 100.0 [100.0, 100.0] | 99.4 [98.5, 100.0] | 98.9 [98.7, 99.0] |
| P | 0.8 | 100.0 [100.0, 100.0] | 99.4 [98.5, 100.0] | 98.9 [98.7, 99.0] |
| C | 0.99 | 100.0 [100.0, 100.0] | 97.8 [96.1, 99.2] | 39.4 [37.4, 41.4] |
| C | 0.95 | 100.0 [100.0, 100.0] | 97.8 [96.1, 99.2] | 39.4 [37.4, 41.4] |
| C | 0.9 | 100.0 [100.0, 100.0] | 97.8 [96.1, 99.2] | 39.4 [37.4, 41.4] |
| C | 0.8 | 80.4 [71.6, 88.1] | 67.2 [62.5, 72.2] | 18.1 [16.7, 19.5] |
| PC | 0.99 | 100.0 [100.0, 100.0] | 97.5 [95.7, 99.0] | 42.2 [40.2, 44.1] |
| PC | 0.95 | 100.0 [100.0, 100.0] | 97.5 [95.7, 99.0] | 42.2 [40.2, 44.1] |
| PC | 0.9 | 100.0 [100.0, 100.0] | 97.5 [95.7, 99.0] | 42.2 [40.2, 44.1] |
| PC | 0.8 | 86.3 [78.9, 92.9] | 77.4 [73.3, 81.7] | 22.5 [20.9, 24.1] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): K.knn 1 / 291; R.title_comention 1 / 122

## musique RK CH

any-link candidates / covered slots: subset 104,777 / 275; rest of dev 333,232 / 813.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| P | 0.95 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| P | 0.9 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| P | 0.8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| C | 0.99 | 100.0 [100.0, 100.0] | 99.8 [99.4, 100.0] | 99.1 [99.0, 99.2] |
| C | 0.95 | 97.1 [94.9, 98.9] | 95.6 [94.1, 96.9] | 82.0 [80.4, 83.5] |
| C | 0.9 | 90.2 [86.7, 93.4] | 89.7 [87.6, 91.7] | 63.0 [61.1, 65.1] |
| C | 0.8 | 89.8 [86.2, 93.1] | 89.2 [87.1, 91.2] | 62.5 [60.5, 64.5] |
| PC | 0.99 | 100.0 [100.0, 100.0] | 97.8 [96.7, 98.7] | 98.2 [98.0, 98.3] |
| PC | 0.95 | 96.0 [93.6, 98.1] | 92.5 [90.5, 94.3] | 69.8 [67.8, 71.9] |
| PC | 0.9 | 92.0 [89.0, 95.1] | 87.2 [84.9, 89.3] | 53.9 [51.7, 56.3] |
| PC | 0.8 | 83.3 [79.3, 87.3] | 79.6 [76.9, 82.2] | 35.8 [33.9, 38.0] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): none

## musique RK GP

any-link candidates / covered slots: subset 104,777 / 281; rest of dev 333,232 / 847.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| P | 0.95 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| P | 0.9 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| P | 0.8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 [99.6, 99.7] |
| C | 0.99 | 100.0 [100.0, 100.0] | 99.8 [99.4, 100.0] | 99.1 [99.0, 99.2] |
| C | 0.95 | 97.2 [95.0, 98.9] | 95.6 [94.2, 97.0] | 82.0 [80.4, 83.5] |
| C | 0.9 | 90.0 [86.5, 93.2] | 88.8 [86.7, 90.8] | 62.5 [60.5, 64.5] |
| C | 0.8 | 90.0 [86.5, 93.2] | 88.8 [86.7, 90.8] | 62.5 [60.5, 64.5] |
| PC | 0.99 | 100.0 [100.0, 100.0] | 97.9 [96.8, 98.8] | 98.2 [98.0, 98.3] |
| PC | 0.95 | 96.1 [93.7, 98.2] | 92.4 [90.5, 94.2] | 69.8 [67.8, 71.9] |
| PC | 0.9 | 91.8 [88.8, 94.8] | 87.2 [85.0, 89.3] | 53.9 [51.7, 56.3] |
| PC | 0.8 | 81.1 [76.8, 85.3] | 76.7 [74.0, 79.5] | 32.9 [31.2, 34.9] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): none

## metaqa SRK GP

any-link candidates / covered slots: subset 406,311 / 223; rest of dev 1,207,737 / 628.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 97.3 [94.5, 99.3] | 99.9 [99.8, 99.9] |
| P | 0.95 | 100.0 [100.0, 100.0] | 97.3 [94.5, 99.3] | 99.9 [99.8, 99.9] |
| P | 0.9 | 100.0 [100.0, 100.0] | 97.3 [94.5, 99.3] | 99.9 [99.8, 99.9] |
| P | 0.8 | 100.0 [100.0, 100.0] | 97.3 [94.5, 99.3] | 99.9 [99.8, 99.9] |
| C | 0.99 | 100.0 [100.0, 100.0] | 99.5 [98.9, 100.0] | 93.0 [92.4, 93.4] |
| C | 0.95 | 98.7 [97.3, 100.0] | 99.2 [98.4, 99.8] | 88.2 [87.3, 88.9] |
| C | 0.9 | 92.8 [88.5, 96.4] | 92.4 [90.2, 94.4] | 81.7 [80.7, 82.5] |
| C | 0.8 | 92.8 [88.5, 96.4] | 92.4 [90.2, 94.4] | 81.7 [80.7, 82.5] |
| PC | 0.99 | 100.0 [100.0, 100.0] | 96.7 [93.7, 98.8] | 88.2 [87.3, 88.9] |
| PC | 0.95 | 100.0 [100.0, 100.0] | 96.7 [93.7, 98.8] | 88.2 [87.3, 88.9] |
| PC | 0.9 | 92.8 [88.5, 96.4] | 89.6 [86.3, 92.6] | 81.7 [80.7, 82.6] |
| PC | 0.8 | 92.8 [88.5, 96.4] | 89.6 [86.3, 92.6] | 81.7 [80.7, 82.6] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): R.name_comention+K.knn 12 / 277; R.mention+K.knn 5 / 695

## webqsp SRK GP

any-link candidates / covered slots: subset 4,019,029 / 226; rest of dev 11,768,177 / 624.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 96.0 [90.2, 99.4] | 100.0 [100.0, 100.0] |
| P | 0.95 | 100.0 [100.0, 100.0] | 96.0 [90.2, 99.4] | 100.0 [100.0, 100.0] |
| P | 0.9 | 100.0 [100.0, 100.0] | 96.0 [90.2, 99.4] | 100.0 [100.0, 100.0] |
| P | 0.8 | 100.0 [100.0, 100.0] | 96.0 [90.2, 99.4] | 100.0 [100.0, 100.0] |
| C | 0.99 | 99.6 [98.4, 100.0] | 98.4 [97.2, 99.4] | 51.3 [45.8, 56.0] |
| C | 0.95 | 99.6 [98.4, 100.0] | 98.4 [97.2, 99.4] | 51.3 [45.8, 56.0] |
| C | 0.9 | 99.6 [98.4, 100.0] | 98.4 [97.2, 99.4] | 51.3 [45.8, 56.0] |
| C | 0.8 | 99.6 [98.4, 100.0] | 98.4 [97.2, 99.4] | 51.3 [45.8, 56.0] |
| PC | 0.99 | 99.6 [98.4, 100.0] | 94.4 [88.6, 98.0] | 51.4 [46.0, 56.1] |
| PC | 0.95 | 99.6 [98.4, 100.0] | 94.4 [88.6, 98.0] | 51.4 [46.0, 56.1] |
| PC | 0.9 | 99.6 [98.4, 100.0] | 94.4 [88.6, 98.0] | 51.4 [46.0, 56.1] |
| PC | 0.8 | 99.6 [98.4, 100.0] | 94.4 [88.6, 98.0] | 51.4 [46.0, 56.1] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): R.mention+R.name_comention+K.knn 11 / 246; R.name_comention+K.knn 7 / 379; R.mention+R.name_comention 5 / 471; R.name_comention 2 / 1599

## hotpotqa RK GP

any-link candidates / covered slots: subset 103,423 / 246; rest of dev 199,335 / 782.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| P | 0.95 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| P | 0.9 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| P | 0.8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| C | 0.99 | 100.0 [100.0, 100.0] | 99.2 [98.6, 99.7] | 99.3 [99.0, 99.5] |
| C | 0.95 | 96.3 [93.9, 98.4] | 97.4 [96.3, 98.5] | 83.8 [79.9, 87.0] |
| C | 0.9 | 90.6 [87.1, 94.2] | 92.6 [90.7, 94.4] | 68.7 [61.3, 74.3] |
| C | 0.8 | 80.1 [75.2, 85.0] | 80.9 [78.1, 83.8] | 47.2 [40.8, 52.2] |
| PC | 0.99 | 100.0 [100.0, 100.0] | 97.2 [96.0, 98.3] | 92.2 [89.0, 94.3] |
| PC | 0.95 | 95.5 [93.0, 98.0] | 93.1 [91.2, 94.8] | 69.1 [62.0, 74.7] |
| PC | 0.9 | 91.5 [88.0, 94.8] | 87.0 [84.5, 89.4] | 50.8 [44.0, 56.1] |
| PC | 0.8 | 91.5 [88.0, 94.8] | 87.0 [84.5, 89.4] | 50.8 [44.0, 56.1] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): none

## hotpotqa SRK GP

any-link candidates / covered slots: subset 109,913 / 296; rest of dev 229,625 / 913.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| P | 0.95 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| P | 0.9 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| P | 0.8 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| C | 0.99 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| C | 0.95 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| C | 0.9 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| C | 0.8 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| PC | 0.99 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| PC | 0.95 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| PC | 0.9 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |
| PC | 0.8 | 100.0 [100.0, 100.0] | 99.6 [99.1, 99.9] | 59.1 [52.8, 63.5] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): K.knn 3 / 7144; R.name_comention 1 / 29170

## 2wiki RK GP

any-link candidates / covered slots: subset 61,024 / 340; rest of dev 176,396 / 1,039.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 99.4 [98.5, 100.0] | 99.4 [98.9, 99.8] | 95.0 [94.6, 95.3] |
| P | 0.95 | 95.0 [92.4, 97.3] | 97.3 [96.2, 98.2] | 65.7 [63.6, 67.5] |
| P | 0.9 | 95.0 [92.4, 97.3] | 97.3 [96.2, 98.2] | 65.7 [63.6, 67.5] |
| P | 0.8 | 95.0 [92.4, 97.3] | 97.3 [96.2, 98.2] | 65.7 [63.6, 67.5] |
| C | 0.99 | 99.7 [99.1, 100.0] | 99.7 [99.3, 100.0] | 78.0 [77.0, 78.9] |
| C | 0.95 | 95.3 [93.0, 97.4] | 93.8 [92.3, 95.4] | 56.2 [54.9, 57.3] |
| C | 0.9 | 95.3 [93.0, 97.4] | 93.8 [92.3, 95.4] | 56.2 [54.9, 57.3] |
| C | 0.8 | 80.9 [76.7, 84.9] | 78.5 [75.9, 81.2] | 36.0 [34.9, 37.1] |
| PC | 0.99 | 99.7 [99.0, 100.0] | 96.9 [95.8, 97.9] | 67.2 [65.5, 68.8] |
| PC | 0.95 | 97.6 [96.0, 99.1] | 95.2 [93.8, 96.5] | 48.4 [46.4, 50.3] |
| PC | 0.9 | 94.1 [91.3, 96.5] | 93.9 [92.4, 95.4] | 43.3 [41.0, 45.4] |
| PC | 0.8 | 87.4 [83.6, 90.7] | 86.8 [84.6, 88.8] | 30.9 [29.4, 32.3] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): R.name_comention 22 / 51733; K.knn 6 / 8833

## 2wiki SRK GP

any-link candidates / covered slots: subset 65,226 / 363; rest of dev 186,794 / 1,082.

S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:

| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |
|---|---:|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 99.4 [98.9, 99.8] | 94.9 [94.5, 95.2] |
| P | 0.95 | 98.3 [96.9, 99.5] | 98.9 [98.2, 99.5] | 67.6 [65.8, 69.3] |
| P | 0.9 | 91.5 [88.4, 94.4] | 94.5 [93.2, 95.9] | 40.2 [37.7, 42.6] |
| P | 0.8 | 91.5 [88.4, 94.4] | 94.5 [93.2, 95.9] | 40.2 [37.7, 42.6] |
| C | 0.99 | 99.2 [98.1, 100.0] | 98.8 [98.1, 99.4] | 61.3 [59.7, 62.9] |
| C | 0.95 | 97.8 [96.1, 99.2] | 98.2 [97.3, 98.9] | 50.2 [48.2, 52.2] |
| C | 0.9 | 91.5 [88.4, 94.4] | 94.5 [93.2, 95.9] | 40.2 [37.7, 42.6] |
| C | 0.8 | 91.5 [88.4, 94.4] | 94.5 [93.2, 95.9] | 40.2 [37.7, 42.6] |
| PC | 0.99 | 99.7 [99.1, 100.0] | 98.3 [97.5, 99.1] | 51.8 [49.7, 53.9] |
| PC | 0.95 | 98.1 [96.5, 99.4] | 97.9 [97.0, 98.7] | 47.0 [44.6, 49.4] |
| PC | 0.9 | 91.5 [88.4, 94.4] | 94.5 [93.2, 95.9] | 40.2 [37.7, 42.6] |
| PC | 0.8 | 91.5 [88.4, 94.4] | 94.5 [93.2, 95.9] | 40.2 [37.7, 42.6] |

P (S protected) at RHO 0.95, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): R.name_comention 6 / 50864; K.knn 6 / 8683
