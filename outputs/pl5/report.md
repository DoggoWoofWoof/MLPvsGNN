### Gates and cost

| dataset | G-R (PL2's, PL3's and PL4's cells rebuilt: first 400 dev questions, PL4's 150) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q, 799,176 elements, mismatches 0/0) | 3, 102 s, 0.28 GB |
| squad | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 7,941 s, 0.12 GB |
| hotpotqa | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 4,376 s, 15.26 GB |
| 2wiki | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 6,850 s, 17.48 GB |
| musique | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 9,740 s, 0.47 GB |
| metaqa | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 1,982 s, 0.4 GB |
| webqsp | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 9,847 s, 6.33 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 552, squad PASS 552; G-X5 PASS, PASS); code-path sha256 `62a27dbb...`.

### Dev reading (the rule's own numbers, dev carve, frozen lists): PL4's chosen cell (the incumbent, reading `prev`) against the grid of all 6,078 cells

| dataset | dev q | PL4 R95 (prev) | R95 (all cells) | PL4 R98 | R98 (all cells) | R95 of the PL5 cells only | prev pick re-read = PL4's filed pick |
|---|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` (0.9955 at 75.0) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b25+STRUCT:h1_cW~cs+top50` (0.9713 at 73.7) | True |
| hotpotqa | 1508 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b25+STRUCT:h1_cW~cs+top100` (0.9602 at 77.3) | True |
| 2wiki | 1496 | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (0.9626 at 2,215.1, SHORT) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (0.9626 at 2,215.1, SHORT) | `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` (0.9525 at 1,269.5) | True |
| musique | 1534 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.8898 at 2,464.3, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.8898 at 2,464.3, SHORT) | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.8898 at 2,464.3, SHORT) | True |
| metaqa | 1497 | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (0.7882 at 2,276.8, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (0.8759 at 2,094.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (0.8759 at 2,094.2, SHORT) | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (0.8759 at 2,094.2, SHORT) | True |

### The chosen cell at 0.95 (R95) and its confirmation on eval: PL4's against the grid of all cells (and the PL5 cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL4 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag all | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag PL5 only | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` | 0.9955 | 75.0 | 0.9885 (75.0) | 0.9823 (75.0) |
| squad PL4 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| squad all | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| squad PL5 only | `rrf1000_b25+STRUCT:h1_cW~cs+top50` | 0.9713 | 73.7 | 0.9721 (73.2) | not read |
| hotpotqa PL4 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa all | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa PL5 only | `rrf1000_b25+STRUCT:h1_cW~cs+top100` | 0.9602 | 77.3 | 0.9616 (77.9) | 0.9633 (77.6) |
| 2wiki PL4 | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki all | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki PL5 only | `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` | 0.9525 | 1,269.5 | 0.9519 (1,265.2) | 0.9719 (1,280.4) |
| musique PL4 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | not read |
| musique all | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| musique PL5 only | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| metaqa PL4 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa PL5 only | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.7882 | 2,276.8 | 0.7785 (2,277.0) SHORT | not read |
| webqsp PL4 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |
| webqsp all | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |
| webqsp PL5 only | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |

### The chosen cell at 0.98 (R98) and its confirmation on eval: PL4's against the grid of all cells (and the PL5 cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL4 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag all | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag PL5 only | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` | 0.9955 | 75.0 | 0.9885 (75.0) | 0.9823 (75.0) |
| squad PL4 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| squad all | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| squad PL5 only | `rrf1000_b25+FULL:h3_bf_v2000~s10-cs-vr+top50` | 0.9846 | 75.0 | 0.9816 (75.0) | not read |
| hotpotqa PL4 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa all | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa PL5 only | `rrf1000_b100+STRUCT:h1_cW~cs+top400` | 0.9801 | 162.2 | 0.9783 (166.4) SHORT | 0.9754 (167.3) SHORT |
| 2wiki PL4 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| 2wiki all | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| 2wiki PL5 only | `rrf1000_b1600+FULL:h2_bf_v1000~cs-fu` (SHORT) | 0.9572 | 2,412.8 | 0.9568 (2,405.5) SHORT | 0.9758 (2,435.7) SHORT |
| musique PL4 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | not read |
| musique all | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| musique PL5 only | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| metaqa PL4 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa PL5 only | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.7882 | 2,276.8 | 0.7785 (2,277.0) SHORT | not read |
| webqsp PL4 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |
| webqsp all | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |
| webqsp PL5 only | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |

### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | not read |
| hotpotqa | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9684 (93.7) | 0.9646 (92.2) |
| 2wiki | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9058 (105.6) | 0.9310 (101.6) |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | not read |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | not read |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | not read |

### The price of the floor at any size, dev (the bound is not applied): the smallest cell with ALL >= 0.95 and >= 0.98, PL4's grid / all cells

| dataset | 0.95: PL4's grid | 0.95: all cells | 0.98: PL4's grid | 0.98: all cells |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7) |
| musique | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) |
| metaqa | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2) | none (best `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2) |
| webqsp | none (best `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1) | none (best `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9257 at 6,070.5) | none (best `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1) | none (best `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9257 at 6,070.5) |

### The highest ALL on dev (the bound is 2,500 mean nodes): PL4's grid / all cells, within the bound and at any size

| dataset | PL4's grid, within the bound | all cells, within the bound | PL4's grid, any size | all cells, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| hotpotqa | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 | `rrf1000_b400+FULL:h3_bf_v5000~cs-fu` 0.9980 at 5,173.9 |
| 2wiki | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9626 at 2,215.1 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9626 at 2,215.1 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7 |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8781 at 2,298.7 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8898 at 2,464.3 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8759 at 2,094.2 | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9257 at 6,070.5 |

### What the PL5 cells bought (dev frontier of all cells minus the frontier of PL2-PL4's cells, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 | 3000 | 4000 | 5000 | 6000 | 7000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9853 / 0.9853 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9927 / 0.9927 (+0.00) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9960 / 0.9960 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9980 / 0.9980 (+0.00) | 0.9987 / 0.9987 (+0.00) | 0.9993 / 0.9993 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| hotpotqa | 0.6127 / 0.6127 (+0.00) | 0.9224 / 0.9224 (+0.00) | 0.9496 / 0.9496 (+0.00) | 0.9682 / 0.9682 (+0.00) | 0.9788 / 0.9788 (+0.00) | 0.9814 / 0.9821 (+0.07) | 0.9834 / 0.9841 (+0.07) | 0.9834 / 0.9841 (+0.07) | 0.9887 / 0.9887 (+0.00) | 0.9901 / 0.9901 (+0.00) | 0.9927 / 0.9927 (+0.00) | 0.9934 / 0.9940 (+0.07) | 0.9940 / 0.9954 (+0.13) | 0.9960 / 0.9960 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9980 / 0.9980 (+0.00) | 0.9980 / 0.9980 (+0.00) |
| 2wiki | 0.3195 / 0.3195 (+0.00) | 0.7172 / 0.7172 (+0.00) | 0.8342 / 0.8342 (+0.00) | 0.8890 / 0.8890 (+0.00) | 0.8984 / 0.8984 (+0.00) | 0.9078 / 0.9078 (+0.00) | 0.9164 / 0.9164 (+0.00) | 0.9298 / 0.9298 (+0.00) | 0.9325 / 0.9325 (+0.00) | 0.9452 / 0.9452 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9626 / 0.9626 (+0.00) | 0.9626 / 0.9626 (+0.00) | 0.9646 / 0.9646 (+0.00) | 0.9646 / 0.9646 (+0.00) | 0.9766 / 0.9766 (+0.00) | 0.9786 / 0.9786 (+0.00) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6349 / 0.6349 (+0.00) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7373 / 0.7373 (+0.00) | 0.7673 / 0.7673 (+0.00) | 0.7947 / 0.7947 (+0.00) | 0.8246 / 0.8455 (+2.09) | 0.8559 / 0.8677 (+1.17) | 0.8781 / 0.8898 (+1.17) | 0.8846 / 0.8944 (+0.98) | 0.8957 / 0.9009 (+0.52) | 0.8990 / 0.9009 (+0.20) | 0.9355 / 0.9355 (+0.00) | 0.9355 / 0.9355 (+0.00) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7308 / 0.7308 (+0.00) | 0.8938 / 0.8938 (+0.00) | 0.8978 / 0.8978 (+0.00) | 0.8998 / 0.8998 (+0.00) | 0.9272 / 0.9272 (+0.00) | 0.9613 / 0.9613 (+0.00) | 0.9646 / 0.9646 (+0.00) | 0.9653 / 0.9653 (+0.00) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6058 / 0.6516 (+4.58) | 0.6603 / 0.6994 (+3.92) | 0.6802 / 0.7213 (+4.11) | 0.6802 / 0.7226 (+4.25) | 0.6855 / 0.7273 (+4.18) | 0.7147 / 0.7857 (+7.10) | 0.7339 / 0.8315 (+9.75) | 0.7571 / 0.8494 (+9.22) | 0.7731 / 0.8759 (+10.29) | 0.7883 / 0.8792 (+9.09) | 0.8056 / 0.8852 (+7.96) | 0.8401 / 0.9018 (+6.17) | 0.8507 / 0.9224 (+7.17) | 0.8540 / 0.9257 (+7.17) |

(each entry: the frontier of PL2-PL4's cells / the frontier of all cells (points bought); `-` = no cell is that small; 3,000 and above are beside the bound)

### What each option bought: R95 of the option's own cells (dev, inside the bound) beside PL4's R95, and its best ALL inside the bound

| dataset | PL4 R95 | budget-following cap (any key) | cosine order (bf) | fused order (bf) | families take turns (bf) | seeds 10/20 | one-hop whole-window walks | 4,096-entry window |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | 0.9911 at 75.0 | 0.9911 at 75.0 | 0.9866 at 75.0 | 0.9866 at 75.0 | 0.9955 at 75.0 | 0.9955 at 75.0 | 0.9866 at 75.0 |
| squad | 0.9519 at 15.0 | 0.9800 at 74.5 | 0.9800 at 74.5 | 0.9793 at 74.5 | 0.9840 at 75.0 | 0.9713 at 74.4 | 0.9713 at 73.7 | 0.9800 at 74.5 |
| hotpotqa | 0.9622 at 77.3 | 0.9595 at 824.5 | 0.9602 at 899.4 | 0.9595 at 824.5 | 0.9602 at 899.4 | 0.9615 at 83.7 | 0.9602 at 77.3 | 0.9735 at 1,990.0 |
| 2wiki | 0.9545 at 908.2 | 0.9525 at 1,269.5 | 0.9512 at 2,314.7 | 0.9525 at 1,269.5 | 0.9539 at 2,315.2 | 0.9392 at 2,306.9 SHORT | 0.9432 at 1,804.3 SHORT | 0.9485 at 2,291.9 SHORT |
| musique | 0.8781 at 2,298.7 SHORT | 0.8898 at 2,464.3 SHORT | 0.8898 at 2,464.3 SHORT | 0.8761 at 2,476.5 SHORT | 0.8898 at 2,464.3 SHORT | 0.8827 at 2,439.4 SHORT | 0.8449 at 2,032.2 SHORT | 0.8781 at 2,372.9 SHORT |
| metaqa | 0.8978 at 2,339.7 SHORT | 0.7882 at 2,276.8 SHORT | 0.6152 at 2,491.0 SHORT | 0.7882 at 2,276.8 SHORT | 0.5845 at 2,237.5 SHORT | 0.6152 at 2,491.0 SHORT | 0.4549 at 1,656.5 SHORT | 0.5157 at 2,219.2 SHORT |
| webqsp | 0.7731 at 2,442.4 SHORT | 0.8759 at 2,094.2 SHORT | 0.8666 at 2,499.5 SHORT | 0.8759 at 2,094.2 SHORT | 0.8427 at 2,064.0 SHORT | 0.8666 at 2,499.5 SHORT | 0.7810 at 1,709.5 SHORT | 0.8374 at 2,459.8 SHORT |

### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

### The call, the old rule as PL3 and PL4 state it (R95 of all cells against R95 of PL2-PL4's cells): CARRY_PL5 (read 7 of 7: 2wiki, hotpotqa, metaqa, multihoprag, musique, squad, webqsp; BELOW 0, ABOVE 1, LEVEL 6)

| dataset | status | NEW (R95, all cells) | NEW ALL (nodes) | OLD (PL4 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | LEVEL (same cell) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | LEVEL (same cell) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | LEVEL | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 0.8846 (2,438.2) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 (2,274.9) | +0.25 [-0.37, +0.83] | 1.072 [1.071, 1.073] |
| metaqa | LEVEL (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| webqsp | ABOVE | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 0.8789 (2,094.6) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 (2,468.1) | +9.85 [+8.18, +11.65] | 0.849 [0.829, 0.870] |

Beside the call (decide nothing): R98 and the PL5-cells-only pick against PL4's R95 / R98.

| dataset | R98 status | R98 NEW ALL (nodes) vs OLD | PL5-only status | PL5-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | LEVEL | 0.9788 (58.0) vs 0.9788 (58.0) | LEVEL | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` | 0.9885 (75.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h1_cW~cs+top50` | 0.9721 (73.2) |
| hotpotqa | LEVEL | 0.9783 (153.5) vs 0.9783 (153.5) | BELOW | `rrf1000_b25+STRUCT:h1_cW~cs+top100` | 0.9616 (77.9) |
| 2wiki | LEVEL | 0.9656 (2,211.2) vs 0.9656 (2,211.2) | BELOW | `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` | 0.9519 (1,265.2) |
| musique | LEVEL | 0.8846 (2,438.2) vs 0.8821 (2,274.9) | LEVEL | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 0.8846 (2,438.2) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8968 (2,335.9) | ABOVE | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` | 0.7785 (2,277.0) |
| webqsp | ABOVE | 0.8789 (2,094.6) vs 0.7804 (2,468.1) | ABOVE | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 0.8789 (2,094.6) |

### The call, the frontier reading (matched node counts inside the bound, declared before any number): CARRY_PL5_F (BELOW 0, ABOVE 2, LEVEL 5)

Per dataset and budget N: the cell of all cells with the highest dev ALL at most N dev nodes against the same for PL2-PL4's cells; eval, frozen lists; paired bootstrap of the ALL difference.

| dataset | frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL | 500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 500 | `rrf1000_b400+STRUCT:h1_c100~cs+top50` (same cell) | `rrf1000_b400+STRUCT:h1_c100~cs+top50` | 0.9965 (446.7) | 0.9965 (446.7) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 1000 | `rrf1000_b100+FULL:h3_bf_v5000_w4096~cs-vr+top800` | `rrf1000_b100+FULL:h3_c100_v5000~cs-vr-eq+top800` | 0.9979 (900.0) | 0.9978 (900.0) | +0.01 [+0.00, +0.03] | LEVEL |
| squad | LEVEL | 1500 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` (same cell) | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` | 0.9982 (1,200.0) | 0.9982 (1,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 2000 | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` (same cell) | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` | 0.9991 (1,599.9) | 0.9991 (1,599.9) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 2500 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 500 | `rrf1000_b100+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b100+FULL:h1_c100~s20+top800` | 0.9854 (460.9) | 0.9854 (460.9) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~cs+top400` (same cell) | `rrf1000_b800+FULL:h1_c100~cs+top400` | 0.9896 (931.2) | 0.9896 (931.2) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1500 | `rrf1000_b400+STRUCT:h3_bf_v1000~cs-fu` | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.9916 (1,279.0) | 0.9923 (1,112.4) | -0.07 [-0.24, +0.11] | LEVEL |
| hotpotqa | LEVEL | 2000 | `rrf1000_b800+STRUCT:h3_bf_v1000~cs-fu` | `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9931 (1,609.2) | 0.9945 (1,883.6) | -0.14 [-0.30, +0.01] | LEVEL |
| hotpotqa | LEVEL | 2500 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` | 0.9918 (2,226.5) | 0.9918 (2,226.5) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 500 | `rrf1000_b200+FULL:h1_c100+top400` (same cell) | `rrf1000_b200+FULL:h1_c100+top400` | 0.9360 (421.6) | 0.9360 (421.6) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1500 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2500 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9656 (2,211.2) | 0.9656 (2,211.2) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | ABOVE | 500 | `rrf1000_b400+FULL:h1_c100+brg100` (same cell) | `rrf1000_b400+FULL:h1_c100+brg100` | 0.7683 (498.1) | 0.7683 (498.1) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | ABOVE | 1000 | `rrf1000_b800+FULL:h1_c100~s10+top200` (same cell) | `rrf1000_b800+FULL:h1_c100~s10+top200` | 0.8213 (986.6) | 0.8213 (986.6) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | ABOVE | 1500 | `rrf1000_b800+FULL:h2_bf_v1000~cs-vr-rr+top800` | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.8618 (1,470.0) | 0.8395 (1,357.1) | +2.23 [+1.53, +2.98] | ABOVE |
| musique | ABOVE | 2000 | `rrf1000_b400+FULL:h2_bf_v2000~cs-vr-rr+top1600` | `rrf1000_b400+FULL:h3_c25_v2000~cs-sp-eq+top1600` | 0.8664 (1,999.5) | 0.8527 (2,000.0) | +1.37 [+0.54, +2.23] | ABOVE |
| musique | ABOVE | 2500 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8846 (2,438.2) | 0.8821 (2,274.9) | +0.25 [-0.29, +0.83] | LEVEL |
| metaqa | LEVEL | 500 | `rrf1000_b100+STRUCT:h2_c25+top400` (same cell) | `rrf1000_b100+STRUCT:h2_c25+top400` | 0.5484 (479.6) | 0.5484 (479.6) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1000 | `equal_rrf_budget_200+STRUCT:h2_c25` (same cell) | `equal_rrf_budget_200+STRUCT:h2_c25` | 0.6899 (763.7) | 0.6899 (763.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1500 | `rrf1000_b25+FS:h2_c100_v5000` (same cell) | `rrf1000_b25+FS:h2_c100_v5000` | 0.7289 (1,473.7) | 0.7289 (1,473.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2000 | `rrf1000_b25+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b25+STRUCT:h3_c25_v2000` | 0.8949 (1,995.3) | 0.8949 (1,995.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2500 | `rrf1000_b400+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | ABOVE | 500 | `dense_top200+FULL:h1_cW~s10-cs+top1600` | `dense_top200+FULL:h1_c100~s20+top400` | 0.7319 (425.7) | 0.6973 (384.7) | +3.46 [+2.00, +4.99] | ABOVE |
| webqsp | ABOVE | 1000 | `dense_top200+STRUCT:h3_bf_v1000~cs-fu+top800` | `rrf1000_b800+FULL:h1_c100~s20+top400` | 0.7811 (1,000.0) | 0.7265 (947.7) | +5.46 [+3.46, +7.58] | ABOVE |
| webqsp | ABOVE | 1500 | `dense_top200+STRUCT:h3_bf_v1000~cs-fu` | `rrf1000_b100+FS:h2_c100_v5000+top1600` | 0.8310 (1,099.0) | 0.7299 (1,486.4) | +10.11 [+8.38, +12.04] | ABOVE |
| webqsp | ABOVE | 2000 | `rrf1000_b25+FULL:h2_bf_v2000~s10-cs-vr` | `dense_top200+FS:h2_c100_v5000~cs-vr` | 0.8443 (1,971.1) | 0.7705 (1,966.1) | +7.39 [+5.32, +9.38] | ABOVE |
| webqsp | ABOVE | 2500 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.8789 (2,094.6) | 0.7804 (2,468.1) | +9.85 [+8.18, +11.78] | ABOVE |

### Beside the calls: the price of the floor at any size (eval; the bound does not apply; decides nothing). Counts for R95any: {'BELOW': 0, 'ABOVE': 0, 'LEVEL': 7}; the any-size frontier (3,000-7,000): {'BELOW': 0, 'ABOVE': 1, 'LEVEL': 6}

| dataset | floor | status | NEW cell | NEW ALL (nodes) | OLD cell | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.95 | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| multihoprag | 0.98 | LEVEL (same cell) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | 0.95 | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | 0.98 | LEVEL (same cell) | `rrf1000_b50+KNN:h1_c100+top25` | 0.9818 (63.0) | `rrf1000_b50+KNN:h1_c100+top25` | 0.9818 (63.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | 0.95 | LEVEL (same cell) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | 0.98 | LEVEL (same cell) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9783 (153.5) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9783 (153.5) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | 0.95 | LEVEL (same cell) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | 0.98 | LEVEL (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` | 0.9785 (6,275.0) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` | 0.9785 (6,275.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | 0.95 | LEVEL (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | 0.98 | LEVEL (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| metaqa | 0.95 | LEVEL (same cell) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 (4,972.8) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 (4,972.8) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| metaqa | 0.98 | LEVEL (same cell) | `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 (6,259.3) | `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 (6,259.3) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| webqsp | 0.95 | LEVEL | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9175 (6,079.5) | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` | 0.8390 (6,029.5) | +7.85 [+6.39, +9.45] | 1.008 [1.007, 1.010] |
| webqsp | 0.98 | LEVEL | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9175 (6,079.5) | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` | 0.8390 (6,029.5) | +7.85 [+6.32, +9.25] | 1.008 [1.007, 1.010] |

| dataset | any-size frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL | 3000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 4000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 5000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 6000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 7000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 3000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 4000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 5000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 6000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 7000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 3000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` | 0.9937 (2,963.6) | 0.9937 (2,963.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 4000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` | 0.9937 (2,963.6) | 0.9937 (2,963.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 5000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` | 0.9937 (2,963.6) | 0.9937 (2,963.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 6000 | `rrf1000_b400+FULL:h3_bf_v5000~cs-fu` | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` | 0.9946 (5,178.2) | 0.9951 (5,454.6) | -0.05 [-0.18, +0.07] | LEVEL |
| hotpotqa | LEVEL | 7000 | `rrf1000_b400+FULL:h3_bf_v5000~cs-fu` | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` | 0.9946 (5,178.2) | 0.9951 (5,454.6) | -0.05 [-0.18, +0.07] | LEVEL |
| 2wiki | LEVEL | 3000 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9656 (2,211.2) | 0.9656 (2,211.2) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 4000 | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9675 (3,055.3) | 0.9675 (3,055.3) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 5000 | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9675 (3,055.3) | 0.9675 (3,055.3) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 6000 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-rk` (same cell) | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-rk` | 0.9765 (5,264.5) | 0.9765 (5,264.5) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 7000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` | 0.9785 (6,275.0) | 0.9785 (6,275.0) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 3000 | `rrf1000_b1600+FULL:h2_bf_v2000~s10-cs-vr` | `rrf1000_b1600+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8904 (2,950.2) | 0.8887 (2,703.3) | +0.17 [-0.50, +0.79] | LEVEL |
| musique | LEVEL | 4000 | `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` | `rrf1000_b1600+FULL:h3_c25_v2000~cs-sp` | 0.8945 (2,980.2) | 0.8883 (3,103.9) | +0.62 [+0.04, +1.16] | ABOVE |
| musique | LEVEL | 5000 | `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` | `rrf1000_b400+FULL:h2_c100_v5000~cs-vr-rr` | 0.8945 (2,980.2) | 0.8957 (4,956.7) | -0.12 [-0.99, +0.66] | LEVEL |
| musique | LEVEL | 6000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 7000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 3000 | `rrf1000_b800+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b800+STRUCT:h3_c25_v2000` | 0.8999 (2,699.3) | 0.8999 (2,699.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 4000 | `dense_top200+STRUCT:h3_c25_v4000` (same cell) | `dense_top200+STRUCT:h3_c25_v4000` | 0.9244 (3,600.3) | 0.9244 (3,600.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 5000 | `rrf1000_b25+STRUCT:h3_c100_v5000` (same cell) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 (4,972.8) | 0.9582 (4,972.8) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 6000 | `rrf1000_b800+STRUCT:h3_c100_v5000` (same cell) | `rrf1000_b800+STRUCT:h3_c100_v5000` | 0.9599 (5,601.9) | 0.9599 (5,601.9) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 7000 | `rrf1000_b1600+STRUCT:h3_c100_v5000` (same cell) | `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 (6,259.3) | 0.9615 (6,259.3) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | ABOVE | 3000 | `rrf1000_b800+STRUCT:h3_bf_v2000~cs-fu` | `rrf1000_b800+FS:h2_c100_v5000` | 0.8802 (2,520.2) | 0.7957 (2,948.2) | +8.45 [+6.85, +10.11] | ABOVE |
| webqsp | ABOVE | 4000 | `rrf1000_b1600+STRUCT:h2_bf_v2000~cs-fu` | `rrf1000_b800+FULL:h2_c100_v5000~s20-cs-sp` | 0.8736 (3,247.0) | 0.8084 (3,956.0) | +6.52 [+4.66, +8.25] | ABOVE |
| webqsp | ABOVE | 5000 | `rrf1000_b800+FULL:h2_bf_v5000~cs-fu` | `rrf1000_b25+FULL:h3_c100_v5000~cs-sp-eq` | 0.8975 (4,976.6) | 0.8277 (4,986.8) | +6.99 [+5.26, +8.58] | ABOVE |
| webqsp | ABOVE | 6000 | `rrf1000_b800+STRUCT:h3_bf_v5000~cs-fu` | `rrf1000_b800+FULL:h3_c100_v5000~cs-sp-eq` | 0.9168 (5,488.5) | 0.8383 (5,433.3) | +7.85 [+6.25, +9.45] | ABOVE |
| webqsp | ABOVE | 7000 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` | 0.9175 (6,079.5) | 0.8390 (6,029.5) | +7.85 [+6.39, +9.25] | ABOVE |

### The 2,500 bound on e3v lists (selection reads the frozen dev lists; this is the same cell re-run on the eval carve's e3v lists)

| dataset | cell (R95, all cells) | nodes: dev frozen | eval frozen | eval e3v | e3v over the bound |
|---|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 50.0 | 50.0 | 50.0 | within |
| squad | `rrf1000_b15` | 15.0 | 15.0 | not read | - |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 77.3 | 77.9 | 77.5 | within |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | 908.2 | 917.0 | 916.1 | within |
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 2,464.3 | 2,438.2 | not read | - |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 2,339.7 | 2,335.9 | not read | - |
| webqsp | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 2,094.2 | 2,094.6 | not read | - |

### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps or the budget) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 300 | 36 (0.120) | 39 | 1.08 | 0 (q 0) | 39 (q 36) | 0 (q 0) | 0 | 2,462.3 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 300 | 31 (0.103) | 81 | 2.61 | 0 (q 0) | 81 (q 31) | 0 (q 0) | 0 | 2,093.5 |

### D2: which stage cut the C golds (post-hoc; labels read; the cell's walk traced hop by hop)

| dataset | cell | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 1 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 39 | 0 | 18 | 0 | 0 | 9 | 0 | 12 | True |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 373 | 0 | 15 | 12 | 0 | 0 | 264 | 82 | True |
| webqsp | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 81 | 2 | 35 | 5 | 0 | 19 | 16 | 4 | True |

