# L2D: two-link depth of the unlinked next-hop golds (declared in docs/L2D_TWO_LINK_DEPTH_DIAG.md)

## Call on musique/s1fit: SIGNAL (MATCHED_LIFT and MATCHED_AUC)

The two-link term is declared in its own file now (LTH section 7's next form): each anchor's term on the rows two links from it, with the middle count and the two links' families as its inputs; its lists and columns are built on the host's CPU so its screen can start when LTH's verdict lands.

Rule: SIGNAL iff share2 >= 0.3 and (matched lift2 >= 1.5 or matched AUC of the middle count >= 0.65), with at least 200 A cases.

## Depth from the anchor

| carve | A cases | targets | linked | unlinked | at 2 | at 3 | far | lift2 | matched lift2 | matched AUC mid | AUC mid | AUC rrf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| musique/s1fit | 3022 | 3798 | 0.3323 | 2536 | 0.5690 | 0.3352 | 0.0958 | 2.431 | 2.420 | 0.6697 | 0.6719 | 0.8843 |
| 2wiki/fit | 4354 | 5661 | 0.6119 | 2197 | 0.1748 | 0.4215 | 0.4037 | 0.502 | 0.448 | 0.4074 | 0.4429 | 0.5482 |
| hotpotqa/fit | 2830 | 2830 | 0.7852 | 608 | 0.5641 | 0.2204 | 0.2155 | 1.430 | 1.251 | 0.6119 | 0.6491 | 0.8406 |

## Unlinked candidates and the two-link set

| carve | unlinked candidates at 2 | at 3 | far | two-link rows per case (median, p90) | unlinked per case (median) | mean degree (median) | two-link targets with a gold middle | with a top-five middle |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| musique/s1fit | 0.2341 | 0.4964 | 0.2695 | 1059.0, 2615.2 | 5018.0 | 32.89 | 0.0665 | 0.0949 |
| 2wiki/fit | 0.3479 | 0.1693 | 0.4827 | 314.0, 722.4 | 1086.0 | 4.46 | 0.0026 | 0.0208 |
| hotpotqa/fit | 0.3944 | 0.2273 | 0.3783 | 225.5, 570.2 | 685.0 | 5.39 | 0.0000 | 0.1312 |

## Middle rows of the two-link rows (shares of the unlinked rows; lift = targets over candidates)

| carve | 1-1 targets, cands, lift | 2-3 targets, cands, lift | 4+ targets, cands, lift |
|---|---|---|---|
| musique/s1fit | 0.1684, 0.0783, 2.149 | 0.1751, 0.0507, 3.456 | 0.2256, 0.1051, 2.147 |
| 2wiki/fit | 0.0756, 0.2054, 0.368 | 0.0583, 0.1057, 0.551 | 0.0410, 0.0367, 1.115 |
| hotpotqa/fit | 0.1974, 0.2283, 0.864 | 0.2303, 0.1312, 1.755 | 0.1365, 0.0349, 3.913 |

## Fixed orders of the unlinked candidates (the unlinked targets' ranks)

| carve | order | ranked | T2 | T10 | median | MRR |
|---|---|---:|---:|---:|---:|---:|
| musique/s1fit | rrf | 2536 | 0.0974 | 0.2756 | 49.0 | 0.1239 |
| musique/s1fit | two_first | 2536 | 0.1289 | 0.2571 | 192.0 | 0.1392 |
| musique/s1fit | mid | 2536 | 0.0177 | 0.0615 | 544.0 | 0.0301 |
| 2wiki/fit | rrf | 2197 | 0.1124 | 0.3136 | 310.0 | 0.1354 |
| 2wiki/fit | two_first | 2197 | 0.0191 | 0.0492 | 509.0 | 0.0256 |
| 2wiki/fit | mid | 2197 | 0.0118 | 0.0250 | 496.0 | 0.0173 |
| hotpotqa/fit | rrf | 608 | 0.2056 | 0.4457 | 14.0 | 0.2290 |
| hotpotqa/fit | two_first | 608 | 0.1283 | 0.2352 | 57.0 | 0.1377 |
| hotpotqa/fit | mid | 608 | 0.0724 | 0.1842 | 75.0 | 0.0924 |

## Each family's links alone, and from the nearest top-five row

| carve | links | linked | unlinked at 2 | lift2 |
|---|---|---:|---:|---:|
| musique/s1fit | struct only | 0.2114 | 0.4411 | 2.560 |
| musique/s1fit | ner only | 0.1643 | 0.0901 | 4.341 |
| musique/s1fit | knn only | 0.0461 | 0.0535 | 19.609 |
| musique/s1fit | every family, from the top five | 0.5047 | 0.3786 (of all targets) | - |
| 2wiki/fit | struct only | 0.5934 | 0.2068 | 0.638 |
| 2wiki/fit | ner only | 0.2197 | 0.0765 | 2.745 |
| 2wiki/fit | knn only | 0.0765 | 0.0417 | 4.814 |
| 2wiki/fit | every family, from the top five | 0.7502 | 0.1056 (of all targets) | - |
| hotpotqa/fit | struct only | 0.7307 | 0.5709 | 1.590 |
| hotpotqa/fit | ner only | 0.2191 | 0.1032 | 3.733 |
| hotpotqa/fit | knn only | 0.1399 | 0.0982 | 6.621 |
| hotpotqa/fit | every family, from the top five | 0.8357 | 0.1081 (of all targets) | - |

## By musique hop

| carve | hop | A cases | linked | unlinked | share2 | lift2 |
|---|---|---:|---:|---:|---:|---:|
| musique/s1fit | 2hop | 2017 | 0.4100 | 1190 | 0.6891 | 2.957 |
| musique/s1fit | 3hop | 805 | 0.2683 | 949 | 0.5100 | 2.107 |
| musique/s1fit | 4hop | 200 | 0.1798 | 397 | 0.3501 | 1.646 |
