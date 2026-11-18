# HFO: the hub family H against the gold links

Declared in docs/HFO_HUB_FAMILY_ORACLE_CHECK.md. Call (s1eval): **ADOPT_H** at k = 5. Next: the transfer step (section 7), declared in its own section before it runs.

| k | P1 diff on MuSiQue CH | P2 gain H | P3 others | gain D |
|---|---|---|---|---|
| 3 | [0.0028, 0.0, 0.0056] - | 0.0145 - | musique/GP LEVEL, 2wiki/GP LEVEL, hotpotqa/GP LEVEL | 0.0118 |
| 5 | [0.006, 0.0025, 0.0096] ABOVE | 0.0208 ok | musique/GP LEVEL, 2wiki/GP LEVEL, hotpotqa/GP LEVEL | 0.0148 |
| 10 | [0.0221, 0.0164, 0.0276] ABOVE | 0.0424 ok | musique/GP ABOVE, 2wiki/GP LEVEL, hotpotqa/GP LEVEL | 0.0203 |

## musique (RK, union 3,387,597 pairs)

Budget (new partners per passage): k=3: H 2.7879, D 3.0; k=5: H 4.6444, D 5.0; k=10: H 9.2835, D 10.0

**s1sel GP**: 3,783 slots, union links 1,120 (29.6%); alone W2_S 626, F0T 731, K 231

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 30.2 | 30.2 | 23 | 21 | +0.05 [-0.08, +0.18] | LEVEL |
| 5 | 30.5 | 30.3 | 33 | 26 | +0.19 [+0.00, +0.37] | LEVEL |
| 10 | 31.2 | 30.7 | 59 | 41 | +0.48 [+0.19, +0.78] | ABOVE |

**s1sel CH**: 2,530 slots, union links 1,083 (42.8%); alone W2_S 603, F0T 718, K 230

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 43.4 | 43.3 | 15 | 13 | +0.08 [-0.08, +0.24] | LEVEL |
| 5 | 43.8 | 43.5 | 25 | 17 | +0.32 [+0.08, +0.59] | ABOVE |
| 10 | 44.6 | 43.9 | 46 | 27 | +0.75 [+0.36, +1.17] | ABOVE |

**s1eval GP**: 5,962 slots, union links 2,005 (33.6%); alone W2_S 1,094, F0T 1,438, K 434

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 34.7 | 34.5 | 62 | 55 | +0.12 [-0.08, +0.32] | LEVEL |
| 5 | 35.2 | 34.9 | 91 | 74 | +0.29 [+0.00, +0.56] | LEVEL |
| 10 | 37.2 | 35.7 | 215 | 123 | +1.54 [+1.12, +1.97] | ABOVE |

**s1eval CH**: 3,987 slots, union links 1,982 (49.7%); alone W2_S 1,082, F0T 1,427, K 434

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 51.2 | 50.9 | 58 | 47 | +0.28 [+0.00, +0.56] | LEVEL |
| 5 | 51.8 | 51.2 | 83 | 59 | +0.60 [+0.25, +0.96] | ABOVE |
| 10 | 53.9 | 51.7 | 169 | 81 | +2.21 [+1.64, +2.76] | ABOVE |

## 2wiki (SRK, union 126,635,390 pairs)

Budget (new partners per passage): k=3: H 2.6916, D 3.0; k=5: H 4.485, D 5.0; k=10: H 8.9672, D 10.0

**s1sel GP**: 3,131 slots, union links 1,444 (46.1%); alone W2_S 1,313, F0T 743, K 255, S 1,355

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 46.2 | 46.2 | 1 | 1 | +0.00 [+0.00, +0.00] | LEVEL |
| 5 | 46.2 | 46.2 | 3 | 1 | +0.06 [+0.00, +0.16] | LEVEL |
| 10 | 46.2 | 46.2 | 4 | 2 | +0.06 [+0.00, +0.16] | LEVEL |

**s1eval GP**: 26,331 slots, union links 12,066 (45.8%); alone W2_S 10,744, F0T 6,895, K 2,805, S 11,234

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 46.0 | 46.0 | 42 | 36 | +0.02 [-0.01, +0.05] | LEVEL |
| 5 | 46.0 | 46.0 | 51 | 47 | +0.02 [-0.02, +0.05] | LEVEL |
| 10 | 46.1 | 46.1 | 68 | 60 | +0.03 [-0.01, +0.07] | LEVEL |

## hotpotqa (SRK, union 66,773,876 pairs)

Budget (new partners per passage): k=3: H 2.6745, D 3.0; k=5: H 4.457, D 5.0; k=10: H 8.9118, D 10.0

**s1sel GP**: 1,508 slots, union links 1,211 (80.3%); alone W2_S 913, F0T 575, K 296, S 1,205

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 80.3 | 80.3 | 0 | 0 | +0.00 [+0.00, +0.00] | LEVEL |
| 5 | 80.3 | 80.4 | 0 | 1 | -0.07 [-0.20, +0.00] | LEVEL |
| 10 | 80.4 | 80.4 | 1 | 2 | -0.07 [-0.27, +0.13] | LEVEL |

**s1eval GP**: 7,405 slots, union links 5,925 (80.0%); alone W2_S 4,294, F0T 2,765, K 1,486, S 5,919

| k | U+H % | U+D % | H new | D new | diff (pts) | call |
|---|---|---|---|---|---|---|
| 3 | 80.1 | 80.1 | 6 | 7 | -0.01 [-0.08, +0.04] | LEVEL |
| 5 | 80.2 | 80.2 | 11 | 11 | +0.00 [-0.08, +0.08] | LEVEL |
| 10 | 80.2 | 80.3 | 16 | 19 | -0.04 [-0.15, +0.05] | LEVEL |
