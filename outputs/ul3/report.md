# UL3 part A: the one-hop screen, report

Declared in `docs/UL3_LINK_TIGHTENING.md`. Read: multihoprag, musique, metaqa, webqsp, hotpotqa, 2wiki. Reach and price retained are shares of the any-link graph's covered slots and candidates on s1eval, points with 95% intervals over questions, in percent. The keep-sets are chosen on the dev carve; the call fixes RHO = 0.95, SCREEN_IN = reach retained >= 92 and price retained <= 50 (point values).

## The universal read (a variant passes iff every judged cell is SCREEN_IN)

The call is A.1's (the S cells are never removed; section 12 of the page); the same keep rule with S not protected, as first declared, is beside.

| variant | call (S protected) | cells SCREEN_IN | cells | beside: S not protected |
|---|---|---:|---:|---|
| P | FAILS_SCREEN | 1 | 9 | FAILS_SCREEN |
| C | FAILS_SCREEN | 2 | 9 | FAILS_SCREEN |
| PC | FAILS_SCREEN | 3 | 9 | FAILS_SCREEN |

## multihoprag RK GP

any-link candidates / covered slots: s1eval 265,876 / 2,350; dev 49,671 / 425.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 99.8 [99.2, 100.0] | 99.6 [99.3, 99.8] | 99.2 [99.2, 99.3] | +0.006 | - |
| P | 0.95 | 99.8 [99.2, 100.0] | 99.6 [99.3, 99.8] | 99.2 [99.2, 99.3] | +0.006 | SCREEN_OUT |
| P | 0.9 | 99.8 [99.2, 100.0] | 99.6 [99.3, 99.8] | 99.2 [99.2, 99.3] | +0.006 | - |
| P | 0.8 | 99.8 [99.2, 100.0] | 99.6 [99.3, 99.8] | 99.2 [99.2, 99.3] | +0.006 | - |
| C | 0.99 | 99.5 [98.8, 100.0] | 99.3 [99.0, 99.6] | 65.0 [64.4, 65.7] | +0.932 | - |
| C | 0.95 | 98.8 [97.7, 99.8] | 97.1 [96.4, 97.8] | 45.4 [44.6, 46.1] | +2.017 | SCREEN_IN |
| C | 0.9 | 98.8 [97.7, 99.8] | 97.1 [96.4, 97.8] | 45.4 [44.6, 46.1] | +2.017 | - |
| C | 0.8 | 80.0 [76.3, 83.7] | 78.2 [76.5, 80.0] | 23.2 [22.5, 23.8] | +4.199 | - |
| PC | 0.99 | 99.5 [98.8, 100.0] | 99.3 [99.0, 99.6] | 66.7 [66.1, 67.3] | +0.864 | - |
| PC | 0.95 | 98.8 [97.7, 99.8] | 97.3 [96.6, 97.9] | 47.8 [47.1, 48.5] | +1.831 | SCREEN_IN |
| PC | 0.9 | 91.5 [88.6, 94.2] | 90.2 [88.8, 91.5] | 27.1 [26.3, 27.8] | +4.127 | - |
| PC | 0.8 | 91.5 [88.6, 94.2] | 90.2 [88.8, 91.5] | 27.1 [26.3, 27.8] | +4.127 | - |

No S cell in this version: the two rules are identical.

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): R.name_comention+K.knn q2; R.name_comention+K.knn q5; R.name_comention+K.knn q3; R.name_comention+K.knn q4; K.knn q1; R.name_comention q5; R.name_comention+K.knn q1; R.title_comention q5; R.name_comention q4

## musique RK CH

any-link candidates / covered slots: s1eval 567,894 / 1,984; dev 438,009 / 1,088.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 98.2 [97.6, 98.8] | 99.5 [99.4, 99.6] | -0.009 | - |
| P | 0.95 | 100.0 [100.0, 100.0] | 98.2 [97.6, 98.8] | 99.5 [99.4, 99.6] | -0.009 | SCREEN_OUT |
| P | 0.9 | 100.0 [100.0, 100.0] | 98.2 [97.6, 98.8] | 99.5 [99.4, 99.6] | -0.009 | - |
| P | 0.8 | 100.0 [100.0, 100.0] | 98.2 [97.6, 98.8] | 99.5 [99.4, 99.6] | -0.009 | - |
| C | 0.99 | 99.8 [99.5, 100.0] | 100.0 [99.8, 100.0] | 99.0 [98.9, 99.0] | +0.007 | - |
| C | 0.95 | 96.0 [94.8, 97.1] | 97.4 [96.8, 98.1] | 86.4 [85.5, 87.3] | +0.089 | SCREEN_OUT |
| C | 0.9 | 96.0 [94.8, 97.1] | 97.4 [96.8, 98.1] | 86.4 [85.5, 87.3] | +0.089 | - |
| C | 0.8 | 89.6 [87.8, 91.4] | 91.7 [90.6, 92.9] | 70.7 [69.5, 71.8] | +0.208 | - |
| PC | 0.99 | 100.0 [100.0, 100.0] | 98.0 [97.4, 98.6] | 99.3 [99.2, 99.3] | -0.009 | - |
| PC | 0.95 | 98.6 [97.9, 99.3] | 97.2 [96.5, 97.9] | 89.3 [88.5, 90.1] | +0.062 | SCREEN_OUT |
| PC | 0.9 | 90.1 [88.4, 91.7] | 91.9 [90.7, 93.1] | 65.5 [64.1, 67.1] | +0.281 | - |
| PC | 0.8 | 82.6 [80.6, 84.8] | 85.1 [83.6, 86.6] | 48.8 [47.4, 50.5] | +0.518 | - |

No S cell in this version: the two rules are identical.

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): R.mention+R.title_comention+K.knn q1; R.mention+R.name_comention+K.knn q2; R.mention+R.name_comention+K.knn q3; R.name_comention+K.knn q4; R.mention+R.name_comention+K.knn q1; R.mention+R.title_comention+K.knn q3; R.name_comention+K.knn q5; R.mention+R.name_comention+K.knn q4; R.mention+R.title_comention q2; R.mention+R.title_comention+K.knn q4; R.mention+R.name_comention q3; R.mention+R.name_comention q5; R.mention+K.knn q5; R.mention+R.name_comention q4 ...

## musique RK GP

any-link candidates / covered slots: s1eval 567,894 / 2,013; dev 438,009 / 1,128.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 98.3 [97.7, 98.8] | 99.5 [99.4, 99.6] | -0.009 | - |
| P | 0.95 | 100.0 [100.0, 100.0] | 98.3 [97.7, 98.8] | 99.5 [99.4, 99.6] | -0.009 | SCREEN_OUT |
| P | 0.9 | 100.0 [100.0, 100.0] | 98.3 [97.7, 98.8] | 99.5 [99.4, 99.6] | -0.009 | - |
| P | 0.8 | 100.0 [100.0, 100.0] | 98.3 [97.7, 98.8] | 99.5 [99.4, 99.6] | -0.009 | - |
| C | 0.99 | 99.8 [99.5, 100.0] | 99.7 [99.4, 99.9] | 99.0 [98.9, 99.0] | +0.005 | - |
| C | 0.95 | 96.0 [94.8, 97.1] | 97.2 [96.5, 97.9] | 86.4 [85.5, 87.3] | +0.088 | SCREEN_OUT |
| C | 0.9 | 90.1 [88.3, 91.8] | 92.8 [91.7, 93.9] | 71.3 [70.1, 72.5] | +0.214 | - |
| C | 0.8 | 90.1 [88.3, 91.8] | 92.8 [91.7, 93.9] | 71.3 [70.1, 72.5] | +0.214 | - |
| PC | 0.99 | 100.0 [100.0, 100.0] | 98.1 [97.4, 98.7] | 99.3 [99.2, 99.3] | -0.009 | - |
| PC | 0.95 | 98.6 [97.9, 99.2] | 96.9 [96.2, 97.7] | 89.3 [88.5, 90.1] | +0.060 | SCREEN_OUT |
| PC | 0.9 | 94.8 [93.5, 96.0] | 94.6 [93.6, 95.6] | 78.4 [77.2, 79.6] | +0.147 | - |
| PC | 0.8 | 80.7 [78.5, 82.8] | 84.5 [83.0, 86.1] | 46.0 [44.6, 47.5] | +0.595 | - |

No S cell in this version: the two rules are identical.

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): R.mention+R.title_comention+K.knn q1; R.mention+R.name_comention+K.knn q2; R.mention+R.name_comention+K.knn q3; R.name_comention+K.knn q4; R.mention+R.name_comention+K.knn q1; R.mention+R.title_comention+K.knn q3; R.name_comention+K.knn q5; R.mention+R.name_comention+K.knn q4; R.mention+R.title_comention q2; R.mention+R.title_comention+K.knn q4; R.mention+R.name_comention q3; R.mention+R.name_comention q5; R.mention+K.knn q5; R.mention+R.name_comention q4 ...

## metaqa SRK GP

any-link candidates / covered slots: s1eval 10,302,674 / 4,912; dev 1,614,048 / 851.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.9 [99.9, 99.9] | +0.000 | - |
| P | 0.95 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.9 [99.9, 99.9] | +0.000 | SCREEN_OUT |
| P | 0.9 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.9 [99.9, 99.9] | +0.000 | - |
| P | 0.8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.9 [99.9, 99.9] | +0.000 | - |
| C | 0.99 | 99.6 [99.2, 100.0] | 99.0 [98.4, 99.5] | 90.2 [90.0, 90.5] | +0.009 | - |
| C | 0.95 | 96.2 [94.9, 97.5] | 95.0 [94.1, 95.7] | 83.2 [82.9, 83.5] | +0.013 | SCREEN_OUT |
| C | 0.9 | 96.2 [94.9, 97.5] | 95.0 [94.1, 95.7] | 83.2 [82.9, 83.5] | +0.013 | - |
| C | 0.8 | 96.2 [94.9, 97.5] | 95.0 [94.1, 95.7] | 83.2 [82.9, 83.5] | +0.013 | - |
| PC | 0.99 | 99.6 [99.2, 100.0] | 99.0 [98.4, 99.5] | 90.5 [90.2, 90.7] | +0.009 | - |
| PC | 0.95 | 96.2 [94.9, 97.5] | 95.0 [94.1, 95.7] | 83.7 [83.4, 84.0] | +0.013 | SCREEN_OUT |
| PC | 0.9 | 96.2 [94.9, 97.5] | 95.0 [94.1, 95.7] | 83.7 [83.4, 84.0] | +0.013 | - |
| PC | 0.8 | 96.2 [94.9, 97.5] | 95.0 [94.1, 95.7] | 83.7 [83.4, 84.0] | +0.013 | - |

**S not protected (as first declared), beside**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.6 [99.6, 99.6] | +0.000 | - |
| P | 0.95 | 95.9 [94.8, 96.9] | 96.3 [95.8, 96.8] | 27.7 [27.4, 28.0] | +0.237 | SCREEN_IN |
| P | 0.9 | 95.9 [94.8, 96.9] | 96.3 [95.8, 96.8] | 27.7 [27.4, 28.0] | +0.237 | - |
| P | 0.8 | 95.9 [94.8, 96.9] | 96.3 [95.8, 96.8] | 27.7 [27.4, 28.0] | +0.237 | - |
| C | 0.99 | 99.6 [99.2, 100.0] | 99.0 [98.4, 99.5] | 45.6 [45.2, 46.0] | +0.111 | - |
| C | 0.95 | 95.5 [94.3, 96.6] | 95.3 [94.6, 96.0] | 17.9 [17.8, 18.1] | +0.411 | SCREEN_IN |
| C | 0.9 | 92.1 [90.3, 93.9] | 91.2 [90.3, 92.2] | 10.9 [10.8, 11.1] | +0.702 | - |
| C | 0.8 | 92.1 [90.3, 93.9] | 91.2 [90.3, 92.2] | 10.9 [10.8, 11.1] | +0.702 | - |
| PC | 0.99 | 99.6 [99.2, 100.0] | 98.9 [98.3, 99.5] | 45.9 [45.5, 46.3] | +0.110 | - |
| PC | 0.95 | 95.5 [94.3, 96.6] | 95.3 [94.5, 95.9] | 18.2 [18.0, 18.3] | +0.405 | SCREEN_IN |
| PC | 0.9 | 92.1 [90.3, 93.9] | 91.2 [90.3, 92.2] | 11.4 [11.3, 11.6] | +0.665 | - |
| PC | 0.8 | 92.1 [90.3, 93.9] | 91.2 [90.3, 92.2] | 11.4 [11.3, 11.6] | +0.665 | - |

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): S q1; S q2; S q3; S q4; S q5; S+R.mention q1; S+R.mention q2; S+R.mention q3; S+R.mention q4; S+R.mention q5; S+R.mention+R.name_comention q1; S+R.mention+R.name_comention q2; S+R.mention+R.name_comention q3; S+R.mention+R.name_comention q4 ...

## webqsp SRK GP

any-link candidates / covered slots: s1eval 15,584,012 / 833; dev 15,787,206 / 850.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.000 | - |
| P | 0.95 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.000 | SCREEN_OUT |
| P | 0.9 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.000 | - |
| P | 0.8 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.000 | - |
| C | 0.99 | 99.4 [98.7, 99.9] | 99.4 [98.7, 99.9] | 55.2 [50.6, 59.5] | +0.009 | - |
| C | 0.95 | 99.4 [98.7, 99.9] | 99.4 [98.7, 99.9] | 55.2 [50.6, 59.5] | +0.009 | SCREEN_OUT |
| C | 0.9 | 99.4 [98.7, 99.9] | 99.4 [98.7, 99.9] | 55.2 [50.6, 59.5] | +0.009 | - |
| C | 0.8 | 99.4 [98.7, 99.9] | 99.4 [98.7, 99.9] | 55.2 [50.6, 59.5] | +0.009 | - |
| PC | 0.99 | 99.4 [98.7, 99.9] | 98.7 [97.7, 99.5] | 55.2 [50.6, 59.5] | +0.008 | - |
| PC | 0.95 | 99.4 [98.7, 99.9] | 98.7 [97.7, 99.5] | 55.2 [50.6, 59.5] | +0.008 | SCREEN_OUT |
| PC | 0.9 | 99.4 [98.7, 99.9] | 98.7 [97.7, 99.5] | 55.2 [50.6, 59.5] | +0.008 | - |
| PC | 0.8 | 99.4 [98.7, 99.9] | 98.7 [97.7, 99.5] | 55.2 [50.6, 59.5] | +0.008 | - |

**S not protected (as first declared), beside**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 99.5 [98.5, 100.0] | 100.0 [100.0, 100.0] | -0.000 | - |
| P | 0.95 | 100.0 [100.0, 100.0] | 99.5 [98.5, 100.0] | 100.0 [100.0, 100.0] | -0.000 | SCREEN_OUT |
| P | 0.9 | 100.0 [100.0, 100.0] | 99.5 [98.5, 100.0] | 100.0 [100.0, 100.0] | -0.000 | - |
| P | 0.8 | 100.0 [100.0, 100.0] | 99.5 [98.5, 100.0] | 100.0 [100.0, 100.0] | -0.000 | - |
| C | 0.99 | 99.4 [98.7, 99.9] | 99.4 [98.7, 99.9] | 55.2 [50.6, 59.5] | +0.009 | - |
| C | 0.95 | 97.3 [95.8, 98.5] | 99.2 [98.4, 99.8] | 32.5 [30.5, 34.4] | +0.022 | SCREEN_IN |
| C | 0.9 | 94.6 [92.3, 96.5] | 97.4 [95.8, 98.6] | 24.0 [22.3, 25.8] | +0.033 | - |
| C | 0.8 | 85.9 [81.6, 90.0] | 92.1 [89.1, 94.5] | 18.3 [16.8, 20.0] | +0.043 | - |
| PC | 0.99 | 99.4 [98.7, 99.9] | 96.2 [93.7, 98.2] | 55.1 [50.5, 59.4] | +0.008 | - |
| PC | 0.95 | 97.3 [95.8, 98.5] | 95.9 [93.5, 98.0] | 32.5 [30.6, 34.5] | +0.021 | SCREEN_IN |
| PC | 0.9 | 94.6 [92.3, 96.5] | 94.1 [91.5, 96.4] | 24.1 [22.4, 25.9] | +0.031 | - |
| PC | 0.8 | 86.6 [82.2, 90.7] | 89.7 [86.2, 92.7] | 18.4 [16.8, 20.0] | +0.042 | - |

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): S q1; S q2; S q3; S q4; S q5; S+R.mention q1; S+R.mention q2; S+R.mention q3; S+R.mention q4; S+R.mention q5; S+R.mention+R.name_comention q1; S+R.mention+R.name_comention q2; S+R.mention+R.name_comention q3; S+R.mention+R.name_comention q4 ...

## hotpotqa RK GP

any-link candidates / covered slots: s1eval 1,747,693 / 4,836; dev 302,758 / 1,028.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 99.8 [99.7, 99.9] | 100.0 [100.0, 100.0] | -0.001 | - |
| P | 0.95 | 95.2 [93.9, 96.5] | 94.8 [94.2, 95.4] | 88.2 [84.0, 92.0] | +0.041 | SCREEN_OUT |
| P | 0.9 | 95.2 [93.9, 96.5] | 94.8 [94.2, 95.4] | 88.2 [84.0, 92.0] | +0.041 | - |
| P | 0.8 | 95.2 [93.9, 96.5] | 94.8 [94.2, 95.4] | 88.2 [84.0, 92.0] | +0.041 | - |
| C | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.000 | - |
| C | 0.95 | 96.7 [95.6, 97.8] | 96.4 [95.9, 96.9] | 63.2 [46.8, 79.4] | +0.291 | SCREEN_OUT |
| C | 0.9 | 90.6 [88.7, 92.3] | 89.2 [88.4, 90.1] | 47.5 [33.4, 61.8] | +0.486 | - |
| C | 0.8 | 90.6 [88.7, 92.3] | 89.2 [88.4, 90.1] | 47.5 [33.4, 61.8] | +0.486 | - |
| PC | 0.99 | 100.0 [100.0, 100.0] | 99.2 [98.9, 99.4] | 95.0 [93.2, 96.6] | +0.024 | - |
| PC | 0.95 | 98.2 [97.3, 98.9] | 97.2 [96.8, 97.7] | 62.2 [46.3, 78.0] | +0.311 | SCREEN_OUT |
| PC | 0.9 | 93.0 [91.4, 94.5] | 91.5 [90.7, 92.3] | 44.5 [31.2, 57.6] | +0.585 | - |
| PC | 0.8 | 87.0 [84.9, 89.0] | 85.3 [84.3, 86.3] | 33.1 [23.0, 43.6] | +0.873 | - |

No S cell in this version: the two rules are identical.

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): R.mention+R.title_comention+K.knn q4; R.mention+R.title_comention+K.knn q1; R.mention+R.title_comention+K.knn q2; R.mention+R.title_comention q4; R.mention+R.name_comention+K.knn q3; R.mention+R.title_comention q5; R.mention+R.title_comention q2; R.mention+R.title_comention q3; R.mention+R.title_comention+K.knn q5; R.mention+R.title_comention+K.knn q3; R.mention+R.name_comention+K.knn q1; R.mention+R.name_comention+K.knn q4; R.mention+R.name_comention+K.knn q2; R.mention+R.name_comention q5 ...

## hotpotqa SRK GP

any-link candidates / covered slots: s1eval 2,053,048 / 5,925; dev 339,538 / 1,209.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| P | 0.95 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | SCREEN_OUT |
| P | 0.9 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| P | 0.8 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| C | 0.99 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| C | 0.95 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | SCREEN_OUT |
| C | 0.9 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| C | 0.8 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| PC | 0.99 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| PC | 0.95 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | SCREEN_OUT |
| PC | 0.9 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| PC | 0.8 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |

**S not protected (as first declared), beside**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 99.7 [99.3, 99.9] | 99.7 [99.6, 99.9] | 52.1 [45.8, 57.3] | +0.528 | - |
| P | 0.95 | 99.7 [99.3, 99.9] | 99.7 [99.6, 99.9] | 52.1 [45.8, 57.3] | +0.528 | SCREEN_OUT |
| P | 0.9 | 99.7 [99.3, 99.9] | 99.7 [99.6, 99.9] | 52.1 [45.8, 57.3] | +0.528 | - |
| P | 0.8 | 99.7 [99.3, 99.9] | 99.7 [99.6, 99.9] | 52.1 [45.8, 57.3] | +0.528 | - |
| C | 0.99 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | - |
| C | 0.95 | 99.7 [99.3, 99.9] | 99.9 [99.8, 100.0] | 52.1 [45.8, 57.3] | +0.529 | SCREEN_OUT |
| C | 0.9 | 92.7 [91.1, 94.1] | 91.4 [90.7, 92.1] | 32.0 [22.1, 42.2] | +1.070 | - |
| C | 0.8 | 82.7 [80.4, 84.8] | 79.9 [78.9, 80.9] | 23.9 [16.2, 31.9] | +1.352 | - |
| PC | 0.99 | 99.7 [99.3, 99.9] | 99.5 [99.3, 99.7] | 52.1 [45.8, 57.3] | +0.526 | - |
| PC | 0.95 | 95.0 [93.8, 96.2] | 94.0 [93.4, 94.6] | 32.9 [22.8, 43.4] | +1.071 | SCREEN_IN |
| PC | 0.9 | 90.8 [89.1, 92.4] | 88.9 [88.1, 89.7] | 27.3 [18.6, 36.2] | +1.304 | - |
| PC | 0.8 | 81.6 [79.3, 83.8] | 79.3 [78.2, 80.3] | 18.3 [12.3, 24.5] | +1.928 | - |

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): S q1; S q2; S q3; S q4; S q5; S+R.mention q1; S+R.mention q2; S+R.mention q3; S+R.mention q4; S+R.mention q5; S+R.name_comention q1; S+R.name_comention q2; S+R.name_comention q3; S+R.name_comention q4 ...

## 2wiki RK GP

any-link candidates / covered slots: s1eval 2,011,310 / 11,522; dev 237,420 / 1,379.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.000 | - |
| P | 0.95 | 96.7 [95.8, 97.6] | 95.5 [95.1, 95.8] | 63.3 [62.5, 64.2] | +0.582 | SCREEN_OUT |
| P | 0.9 | 96.7 [95.8, 97.6] | 95.5 [95.1, 95.8] | 63.3 [62.5, 64.2] | +0.582 | - |
| P | 0.8 | 96.7 [95.8, 97.6] | 95.5 [95.1, 95.8] | 63.3 [62.5, 64.2] | +0.582 | - |
| C | 0.99 | 99.6 [99.3, 99.9] | 99.6 [99.4, 99.7] | 76.9 [76.5, 77.3] | +0.338 | - |
| C | 0.95 | 95.3 [94.1, 96.4] | 95.4 [95.0, 95.8] | 57.5 [57.0, 58.0] | +0.754 | SCREEN_OUT |
| C | 0.9 | 94.9 [93.7, 96.1] | 94.9 [94.5, 95.3] | 56.6 [56.1, 57.1] | +0.773 | - |
| C | 0.8 | 81.1 [78.8, 83.2] | 82.7 [82.0, 83.4] | 38.1 [37.7, 38.6] | +1.338 | - |
| PC | 0.99 | 99.1 [98.5, 99.6] | 98.5 [98.3, 98.7] | 61.1 [60.3, 61.9] | +0.703 | - |
| PC | 0.95 | 95.9 [94.9, 97.0] | 94.9 [94.5, 95.4] | 41.0 [40.2, 41.8] | +1.509 | SCREEN_IN |
| PC | 0.9 | 95.9 [94.9, 97.0] | 94.9 [94.5, 95.4] | 41.0 [40.2, 41.8] | +1.509 | - |
| PC | 0.8 | 89.5 [87.8, 91.0] | 89.5 [89.0, 90.1] | 29.9 [29.4, 30.4] | +2.289 | - |

No S cell in this version: the two rules are identical.

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): R.mention+R.title_comention+K.knn q2; R.mention+R.title_comention+K.knn q4; R.mention+R.title_comention+K.knn q3; R.title_comention+K.knn q1; R.title_comention+K.knn q2; R.mention+R.name_comention+K.knn q5; R.mention+R.title_comention+K.knn q1; R.mention+R.name_comention+K.knn q4; R.mention+R.name_comention+K.knn q1; R.mention+R.title_comention q5; R.mention+R.title_comention q4; R.mention+R.name_comention+K.knn q2; R.mention+R.name_comention+K.knn q3; R.mention+R.title_comention+K.knn q5 ...

## 2wiki SRK GP

any-link candidates / covered slots: s1eval 2,131,551 / 12,061; dev 252,020 / 1,445.

**S protected (A.1, the call)**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 99.2 [98.7, 99.6] | 98.4 [98.1, 98.6] | 69.7 [69.1, 70.4] | +0.464 | - |
| P | 0.95 | 96.1 [95.0, 97.0] | 95.8 [95.4, 96.2] | 36.7 [36.0, 37.4] | +1.820 | SCREEN_IN |
| P | 0.9 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |
| P | 0.8 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |
| C | 0.99 | 99.3 [98.9, 99.7] | 99.0 [98.8, 99.2] | 61.2 [60.6, 61.8] | +0.700 | - |
| C | 0.95 | 98.3 [97.6, 98.9] | 97.9 [97.7, 98.2] | 47.6 [47.0, 48.3] | +1.195 | SCREEN_IN |
| C | 0.9 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |
| C | 0.8 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |
| PC | 0.99 | 99.6 [99.2, 99.9] | 98.5 [98.3, 98.8] | 49.6 [48.9, 50.3] | +1.116 | - |
| PC | 0.95 | 95.3 [94.2, 96.4] | 94.5 [94.0, 94.9] | 35.7 [35.0, 36.4] | +1.858 | SCREEN_IN |
| PC | 0.9 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |
| PC | 0.8 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |

**S not protected (as first declared), beside**

| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |
|---|---:|---|---|---|---|---|
| P | 0.99 | 99.2 [98.7, 99.6] | 98.4 [98.1, 98.6] | 69.7 [69.1, 70.4] | +0.464 | - |
| P | 0.95 | 96.1 [95.0, 97.0] | 95.8 [95.4, 96.2] | 36.7 [36.0, 37.4] | +1.820 | SCREEN_IN |
| P | 0.9 | 90.2 [88.5, 91.7] | 89.8 [89.2, 90.3] | 29.9 [29.2, 30.6] | +2.266 | - |
| P | 0.8 | 90.2 [88.5, 91.7] | 89.8 [89.2, 90.3] | 29.9 [29.2, 30.6] | +2.266 | - |
| C | 0.99 | 99.3 [98.9, 99.7] | 99.0 [98.8, 99.2] | 61.2 [60.6, 61.8] | +0.700 | - |
| C | 0.95 | 98.3 [97.6, 98.9] | 97.9 [97.7, 98.2] | 47.6 [47.0, 48.3] | +1.195 | SCREEN_IN |
| C | 0.9 | 93.8 [92.5, 95.0] | 93.1 [92.7, 93.6] | 35.3 [34.6, 36.0] | +1.853 | - |
| C | 0.8 | 88.6 [87.0, 90.2] | 88.5 [88.0, 89.1] | 28.4 [27.9, 29.0] | +2.392 | - |
| PC | 0.99 | 99.6 [99.2, 99.9] | 98.0 [97.7, 98.2] | 49.5 [48.8, 50.2] | +1.107 | - |
| PC | 0.95 | 95.6 [94.4, 96.6] | 94.8 [94.4, 95.1] | 34.5 [33.8, 35.1] | +1.979 | SCREEN_IN |
| PC | 0.9 | 90.1 [88.6, 91.6] | 89.6 [89.1, 90.1] | 27.1 [26.6, 27.6] | +2.617 | - |
| PC | 0.8 | 82.2 [80.2, 84.0] | 83.2 [82.5, 83.8] | 21.5 [21.1, 21.9] | +3.239 | - |

PC's kept cells at RHO 0.95 (S protected; dev; the S cells first, then in precision order): S q1; S q2; S q3; S q4; S q5; S+R.mention q1; S+R.mention q2; S+R.mention q3; S+R.mention q4; S+R.mention q5; S+R.name_comention q1; S+R.name_comention q2; S+R.name_comention q3; S+R.name_comention q4 ...
