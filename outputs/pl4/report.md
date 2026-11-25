### Gates and cost

| dataset | G-R (PL2's 804 + PL3's 1,470 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q x 2274 cells = 509,376 elements, mismatches 0/0) | 3, 82 s, 0.28 GB |
| squad | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 7,012 s, 0.12 GB |
| hotpotqa | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 4,210 s, 14.95 GB |
| 2wiki | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 7,610 s, 17.35 GB |
| musique | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 9,488 s, 0.64 GB |
| metaqa | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 2,392 s, 0.4 GB |
| webqsp | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 8,308 s, 5.46 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 552, squad PASS 552; G-X4 PASS, PASS); code-path sha256 `aa7c9ac9...`.

### Dev reading (the rule's own numbers, dev carve, frozen lists): PL3's chosen cell (the incumbent) against the grid of all 4,206 cells

| dataset | dev q | PL3 R95 (PL2 U PL3 cells) | R95 (all cells) | PL3 R98 | R98 (all cells) | R95 of the new cells only | PL3's pick re-read = its filed pick |
|---|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` (0.9955 at 58.6) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-rk+top50` (0.9780 at 71.9) | True |
| hotpotqa | 1508 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b25+FULL:h3_c25_v2000~rk+top800` (0.9503 at 825.0) | True |
| 2wiki | 1496 | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b1600+FULL:h1_c100~s20+top800` (0.9606 at 2,078.2, SHORT) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (0.9626 at 2,215.1, SHORT) | `rrf1000_b400+FULL:h3_c25_v2000~rk+top1600` (0.9505 at 2,000.0) | True |
| musique | 1534 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | True |
| metaqa | 1497 | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` (0.8303 at 2,284.0, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | True |

### The chosen cell at 0.95 (R95) and its confirmation on eval: PL3's against the grid of all cells (and the new cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL3 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag all | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag new only | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9955 | 58.6 | 0.9788 (58.0) | 0.9779 (58.2) |
| squad PL3 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT |
| squad all | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT |
| squad new only | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-rk+top50` | 0.9780 | 71.9 | 0.9773 (72.8) | 0.9661 (72.3) |
| hotpotqa PL3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa all | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa new only | `rrf1000_b25+FULL:h3_c25_v2000~rk+top800` | 0.9503 | 825.0 | 0.9568 (825.0) | 0.9611 (825.0) |
| 2wiki PL3 | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki all | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki new only | `rrf1000_b400+FULL:h3_c25_v2000~rk+top1600` | 0.9505 | 2,000.0 | 0.9491 (2,000.0) SHORT | 0.9719 (2,000.0) |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | 0.8775 (2,457.0) SHORT |
| musique all | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| musique new only | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa new only | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` (SHORT) | 0.8303 | 2,284.0 | 0.8262 (2,281.6) SHORT | 0.8400 (2,238.7) SHORT |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp all | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp new only | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |

### The chosen cell at 0.98 (R98) and its confirmation on eval: PL3's against the grid of all cells (and the new cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL3 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag all | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag new only | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| squad PL3 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| squad all | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| squad new only | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9820 | 74.9 | 0.9800 (75.0) SHORT | 0.9698 (75.0) SHORT |
| hotpotqa PL3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa all | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa new only | `rrf1000_b800+FULL:h3_c25_v2000~rk+top800` | 0.9801 | 1,600.0 | 0.9780 (1,600.0) SHORT | 0.9792 (1,600.0) SHORT |
| 2wiki PL3 | `rrf1000_b1600+FULL:h1_c100~s20+top800` (SHORT) | 0.9606 | 2,078.2 | 0.9656 (2,085.7) SHORT | 0.9716 (2,088.6) SHORT |
| 2wiki all | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| 2wiki new only | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | 0.8775 (2,457.0) SHORT |
| musique all | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| musique new only | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa new only | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` (SHORT) | 0.8303 | 2,284.0 | 0.8262 (2,281.6) SHORT | 0.8400 (2,238.7) SHORT |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp all | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp new only | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |

### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | 0.9736 (50.0) |
| hotpotqa | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9684 (93.7) | 0.9646 (92.2) |
| 2wiki | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9058 (105.6) | 0.9310 (101.6) |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | 0.7911 (2,087.2) |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | 0.9278 (2,005.1) |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | 0.6447 (2,124.5) |

### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL3's grid / all cells

| dataset | 0.95: PL3's grid | 0.95: all cells | 0.98: PL3's grid | 0.98: all cells |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | none (best 0.9719 at 6,230.3) | none (best 0.9786 at 6,277.7) |
| musique | none (best 0.9355 at 5,783.4) | none (best 0.9355 at 5,783.4) | none (best 0.9355 at 5,783.4) | none (best 0.9355 at 5,783.4) |
| metaqa | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best 0.9653 at 6,268.2) | none (best 0.9653 at 6,268.2) |
| webqsp | none (best 0.8480 at 6,067.3) | none (best 0.8540 at 6,021.1) | none (best 0.8480 at 6,067.3) | none (best 0.8540 at 6,021.1) |

### The highest ALL on dev (the bound is 2,500 mean nodes): PL3's grid / all cells, within the bound and at any size

| dataset | PL3's grid, within the bound | all cells, within the bound | PL3's grid, any size | all cells, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| hotpotqa | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 |
| 2wiki | `rrf1000_b1600+FULL:h1_c100~s20+top800` 0.9606 at 2,078.2 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9626 at 2,215.1 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9719 at 6,230.3 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7 |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8735 at 2,488.9 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8781 at 2,298.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `rrf1000_b1600+FULL:h3_c100_v5000~vr` 0.8480 at 6,067.3 | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1 |

### What the PL4 cells bought (dev frontier of all cells minus PL3's frontier, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9846 / 0.9853 (+0.07) | 0.9887 / 0.9887 (+0.00) | 0.9920 / 0.9927 (+0.07) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9960 / 0.9960 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9973 / 0.9980 (+0.07) | 0.9987 / 0.9987 (+0.00) | 0.9993 / 0.9993 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| hotpotqa | 0.6127 / 0.6127 (+0.00) | 0.9224 / 0.9224 (+0.00) | 0.9496 / 0.9496 (+0.00) | 0.9682 / 0.9682 (+0.00) | 0.9788 / 0.9788 (+0.00) | 0.9814 / 0.9814 (+0.00) | 0.9834 / 0.9834 (+0.00) | 0.9834 / 0.9834 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9901 / 0.9901 (+0.00) | 0.9927 / 0.9927 (+0.00) | 0.9934 / 0.9934 (+0.00) | 0.9940 / 0.9940 (+0.00) | 0.9960 / 0.9960 (+0.00) |
| 2wiki | 0.3195 / 0.3195 (+0.00) | 0.7172 / 0.7172 (+0.00) | 0.8342 / 0.8342 (+0.00) | 0.8890 / 0.8890 (+0.00) | 0.8984 / 0.8984 (+0.00) | 0.9078 / 0.9078 (+0.00) | 0.9164 / 0.9164 (+0.00) | 0.9298 / 0.9298 (+0.00) | 0.9325 / 0.9325 (+0.00) | 0.9452 / 0.9452 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9606 / 0.9626 (+0.20) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6349 / 0.6349 (+0.00) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7373 / 0.7373 (+0.00) | 0.7673 / 0.7673 (+0.00) | 0.7947 / 0.7947 (+0.00) | 0.8246 / 0.8246 (+0.00) | 0.8449 / 0.8559 (+1.11) | 0.8735 / 0.8781 (+0.46) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7308 / 0.7308 (+0.00) | 0.8938 / 0.8938 (+0.00) | 0.8978 / 0.8978 (+0.00) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6058 / 0.6058 (+0.00) | 0.6603 / 0.6603 (+0.00) | 0.6802 / 0.6802 (+0.00) | 0.6802 / 0.6802 (+0.00) | 0.6855 / 0.6855 (+0.00) | 0.7147 / 0.7147 (+0.00) | 0.7339 / 0.7339 (+0.00) | 0.7571 / 0.7571 (+0.00) | 0.7731 / 0.7731 (+0.00) |

(each entry: PL3's frontier / the frontier of all cells (points bought); `-` = no cell is that small)

### What each option bought: R95 of PL3's cells plus that option's cells (dev), and the best ALL inside the bound among that option's cells alone

| dataset | PL3 R95 | support order | row-rank order | equal hop shares | families take turns | seeds 10/20 with a new option |
|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | = PL3; alone best 1.0000 at 123.2 | = PL3; alone best 1.0000 at 123.2 | = PL3; alone best 1.0000 at 111.5 | = PL3; alone best 1.0000 at 123.2 | = PL3; alone best 1.0000 at 123.4 |
| squad | 0.9519 at 15.0 | = PL3; alone best 0.9993 at 1,940.5 | = PL3; alone best 0.9993 at 1,600.0 | = PL3; alone best 0.9993 at 1,700.0 | = PL3; alone best 0.9993 at 1,700.0 | = PL3; alone best 0.9993 at 1,944.7 |
| hotpotqa | 0.9622 at 77.3 | = PL3; alone best 0.9927 at 2,172.6 | = PL3; alone best 0.9934 at 2,327.3 | = PL3; alone best 0.9947 at 2,464.0 | = PL3; alone best 0.9940 at 2,467.1 | = PL3; alone best 0.9934 at 2,408.3 |
| 2wiki | 0.9545 at 908.2 | = PL3; alone best 0.9579 at 2,288.7 | = PL3; alone best 0.9592 at 2,318.3 | = PL3; alone best 0.9626 at 2,215.1 | = PL3; alone best 0.9592 at 2,318.4 | = PL3; alone best 0.9626 at 2,215.1 |
| musique | 0.8735 at 2,488.9 SHORT | 0.8781 at 2,469.5 SHORT; alone best 0.8781 at 2,469.5 | = PL3; alone best 0.8638 at 2,393.8 | 0.8781 at 2,298.7 SHORT; alone best 0.8781 at 2,298.7 | 0.8781 at 2,298.7 SHORT; alone best 0.8781 at 2,298.7 | = PL3; alone best 0.8709 at 2,263.9 |
| metaqa | 0.8978 at 2,339.7 SHORT | = PL3; alone best 0.8303 at 2,284.0 | = PL3; alone best 0.7862 at 2,284.1 | = PL3; alone best 0.8297 at 2,284.0 | = PL3; alone best 0.8003 at 2,260.9 | = PL3; alone best 0.7428 at 2,286.6 |
| webqsp | 0.7731 at 2,442.4 SHORT | 0.7731 at 2,442.4 SHORT; alone best 0.7731 at 2,442.4 | = PL3; alone best 0.7731 at 2,442.5 | = PL3; alone best 0.7379 at 2,498.1 | = PL3; alone best 0.7624 at 2,288.5 | 0.7731 at 2,442.4 SHORT; alone best 0.7731 at 2,442.4 |

### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

### The call, the old rule as PL3 states it (R95 of all cells against PL3's R95): CARRY_PL4 (read 7 of 7: 2wiki, hotpotqa, metaqa, multihoprag, musique, squad, webqsp; BELOW 0, ABOVE 1, LEVEL 6)

| dataset | status | NEW (R95, all cells) | NEW ALL (nodes) | OLD (PL3 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | LEVEL (same cell) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | LEVEL (same cell) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | ABOVE | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 (2,274.9) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) | +0.33 [-0.33, +0.99] | 0.927 [0.925, 0.928] |
| metaqa | LEVEL (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| webqsp | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 (2,468.1) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |

Beside the call (decide nothing): R98 and the new-cells-only pick against PL3's R95.

| dataset | R98 status | R98 NEW ALL (nodes) vs PL3 | new-only status | new-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | LEVEL | 0.9788 (58.0) vs 0.9788 (58.0) | LEVEL | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9788 (58.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-rk+top50` | 0.9773 (72.8) |
| hotpotqa | LEVEL | 0.9783 (153.5) vs 0.9783 (153.5) | BELOW | `rrf1000_b25+FULL:h3_c25_v2000~rk+top800` | 0.9568 (825.0) |
| 2wiki | LEVEL | 0.9656 (2,211.2) vs 0.9656 (2,085.7) | BELOW | `rrf1000_b400+FULL:h3_c25_v2000~rk+top1600` | 0.9491 (2,000.0) |
| musique | ABOVE | 0.8821 (2,274.9) vs 0.8788 (2,455.1) | ABOVE | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 (2,274.9) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8968 (2,335.9) | ABOVE | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` | 0.8262 (2,281.6) |
| webqsp | LEVEL | 0.7804 (2,468.1) vs 0.7804 (2,468.1) | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 (2,468.1) |

### The call, the frontier reading (matched node counts, proposed on the page before any number): NOT_CARRIED (BELOW 0, ABOVE 0, LEVEL 7)

Per dataset and budget N: the cell of all cells with the highest dev ALL at most N dev nodes against the same for PL3's grid; eval, frozen lists; paired bootstrap of the ALL difference.

| dataset | frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL | 500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 500 | `rrf1000_b400+STRUCT:h1_c100~cs+top50` (same cell) | `rrf1000_b400+STRUCT:h1_c100~cs+top50` | 0.9965 (446.7) | 0.9965 (446.7) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 1000 | `rrf1000_b100+FULL:h3_c100_v5000~cs-vr-eq+top800` | `rrf1000_b400+STRUCT:h3_c100_v5000~cs-vr+brg400` | 0.9978 (900.0) | 0.9974 (793.3) | +0.04 [-0.05, +0.13] | LEVEL |
| squad | LEVEL | 1500 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` (same cell) | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` | 0.9982 (1,200.0) | 0.9982 (1,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 2000 | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top200` | 0.9991 (1,599.9) | 0.9992 (1,740.6) | -0.01 [-0.05, +0.03] | LEVEL |
| squad | LEVEL | 2500 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 500 | `rrf1000_b100+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b100+FULL:h1_c100~s20+top800` | 0.9854 (460.9) | 0.9854 (460.9) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~cs+top400` (same cell) | `rrf1000_b800+FULL:h1_c100~cs+top400` | 0.9896 (931.2) | 0.9896 (931.2) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1500 | `rrf1000_b800+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.9923 (1,112.4) | 0.9923 (1,112.4) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 2000 | `rrf1000_b1600+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9945 (1,883.6) | 0.9945 (1,883.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 2500 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` | 0.9918 (2,226.5) | 0.9918 (2,226.5) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 500 | `rrf1000_b200+FULL:h1_c100+top400` (same cell) | `rrf1000_b200+FULL:h1_c100+top400` | 0.9360 (421.6) | 0.9360 (421.6) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1500 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2500 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` | `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9656 (2,211.2) | 0.9656 (2,085.7) | -0.01 [-0.14, +0.13] | LEVEL |
| musique | LEVEL | 500 | `rrf1000_b400+FULL:h1_c100+brg100` (same cell) | `rrf1000_b400+FULL:h1_c100+brg100` | 0.7683 (498.1) | 0.7683 (498.1) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~s10+top200` (same cell) | `rrf1000_b800+FULL:h1_c100~s10+top200` | 0.8213 (986.6) | 0.8213 (986.6) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 1500 | `rrf1000_b800+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.8395 (1,357.1) | 0.8395 (1,357.1) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 2000 | `rrf1000_b400+FULL:h3_c25_v2000~cs-sp-eq+top1600` | `rrf1000_b1600+FULL:h1_c100~s20+top400` | 0.8527 (2,000.0) | 0.8585 (1,946.3) | -0.58 [-1.57, +0.50] | LEVEL |
| musique | LEVEL | 2500 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8821 (2,274.9) | 0.8788 (2,455.1) | +0.33 [-0.29, +0.95] | LEVEL |
| metaqa | LEVEL | 500 | `rrf1000_b100+STRUCT:h2_c25+top400` (same cell) | `rrf1000_b100+STRUCT:h2_c25+top400` | 0.5484 (479.6) | 0.5484 (479.6) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1000 | `equal_rrf_budget_200+STRUCT:h2_c25` (same cell) | `equal_rrf_budget_200+STRUCT:h2_c25` | 0.6899 (763.7) | 0.6899 (763.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1500 | `rrf1000_b25+FS:h2_c100_v5000` (same cell) | `rrf1000_b25+FS:h2_c100_v5000` | 0.7289 (1,473.7) | 0.7289 (1,473.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2000 | `rrf1000_b25+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b25+STRUCT:h3_c25_v2000` | 0.8949 (1,995.3) | 0.8949 (1,995.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2500 | `rrf1000_b400+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 500 | `dense_top200+FULL:h1_c100~s20+top400` (same cell) | `dense_top200+FULL:h1_c100~s20+top400` | 0.6973 (384.7) | 0.6973 (384.7) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~s20+top400` (same cell) | `rrf1000_b800+FULL:h1_c100~s20+top400` | 0.7265 (947.7) | 0.7265 (947.7) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 1500 | `rrf1000_b100+FS:h2_c100_v5000+top1600` (same cell) | `rrf1000_b100+FS:h2_c100_v5000+top1600` | 0.7299 (1,486.4) | 0.7299 (1,486.4) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 2000 | `dense_top200+FS:h2_c100_v5000~cs-vr` (same cell) | `dense_top200+FS:h2_c100_v5000~cs-vr` | 0.7705 (1,966.1) | 0.7705 (1,966.1) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 2500 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | 0.7804 (2,468.1) | +0.00 [+0.00, +0.00] | LEVEL |

### The 2,500 bound on e3v lists (selection reads the frozen dev lists; this is the same cell re-run on the eval carve's e3v lists)

| dataset | cell (R95, all cells) | nodes: dev frozen | eval frozen | eval e3v | e3v over the bound |
|---|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 50.0 | 50.0 | 50.0 | within |
| squad | `rrf1000_b15` | 15.0 | 15.0 | 15.0 | within |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 77.3 | 77.9 | 77.5 | within |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | 908.2 | 917.0 | 916.1 | within |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 2,298.7 | 2,274.9 | 2,274.8 | within |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 2,339.7 | 2,335.9 | 2,325.6 | within |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 2,442.4 | 2,468.1 | 2,720.1 | OVER_BOUND |

### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 300 | 35 (0.117) | 37 | 1.06 | 0 (q 0) | 37 (q 35) | 0 (q 0) | 0 | 2,301.4 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 300 | 63 (0.210) | 245 | 3.89 | 0 (q 0) | 245 (q 63) | 0 (q 0) | 0 | 2,421.1 |

### D2: which stage cut the C golds (post-hoc; labels read; the cell's walk traced hop by hop)

| dataset | cell | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 37 | 0 | 13 | 6 | 9 | 7 | 2 | True |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 373 | 0 | 15 | 12 | 0 | 264 | 82 | True |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 245 | 126 | 97 | 0 | 0 | 0 | 22 | True |

