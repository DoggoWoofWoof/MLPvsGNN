# BRG6: the learned candidate order read on every question the fit did not use (whole-split read)

Declared 28 Nov 2026 20:30 IST, before any number of this page. Read on five sets; nothing is adopted. Module: `outputs/mp_unified/zbrg6.py` (new; imports `zbrg5.py` and `zbrg5b.py`, neither edited; its self-test runs before any row; sha recorded in section 6).

## 1. Why

BRG5 and its Amendment 1 (`docs/BRG5_LEARNED_ORDER_SCREEN.md`, sections 7 and 9) found that a logistic regression on 12 walk features, fitted per hop on the golds of other questions of the same split, is ABOVE the cell's own candidate order on Prime at an unchanged pool size under all four fits (+3.0 to +4.6), and positive but LEVEL on MAG (+0.0 to +1.0) and WebQSP (+0.3 to +1.8), with nothing on MuSiQue and Amazon. Those reads cover 369 to 384 questions, so the 95 percent interval is about plus or minus 2 points, and six training chunks instead of three changed nothing beyond that noise. Whether MAG's and WebQSP's small differences are real needs more questions, and each split has more: the screens read three chunks of 128 and the fits used three to six more; the rest of the split is unread.

This page reads the one fit that Amendment 1 named the larger (`lr6`, section 8 of the BRG5 page: logistic regression, C = 1, 200 iterations, standardized, per hop, fitted on the chunks at positions 3 to 8 of the set's chunk order) on every chunk of the split outside those six. The fit is not refitted and not tuned on this read.

A positive read here is a result on five sets, not a rule. A rule needs the same recipe unchanged on all ten datasets (MetaQA's cap walk and the four sets that already meet the target are not covered here), a fit that does not depend on questions of the read split, and the user's ruling on whether a fitted scorer sits in the rules path or is the first step of the joint walk-and-rank model that is the stated alternative. The page measures; it does not move the line.

## 2. What runs

For each of Prime, MAG, WebQSP, MuSiQue and Amazon, in the cell of BRG1 (`zbrg1.CELL`), the same walk and the same gates as BRG5: two variants, `control` (the cell's own order) and `lr6` (BRG5 Amendment 1's `lr6` fit, refitted from the same training files). Chunks of 128 questions in the set's chunk order (`zpl3g.chunk_order`); the fit's chunks are the positions 3 to 8 (Prime 16, 2, 0, 17, 11, 5; MAG 18, 12, 16, 7, 9, 13; WebQSP 4, 1, 6, 8, 5, 9; MuSiQue 10, 8, 16, 7, 5, 17; Amazon 9, 1, 10, 6, 2, 8; 768 questions each). The read covers every other chunk: the three screen chunks (positions 0 to 2: seen in BRG5) first, then positions 9 and up in order.

| set | chunks read | screen chunks (seen) | questions read | of which unseen |
| --- | --- | --- | --- | --- |
| Prime | 16 of 22 | 10, 18, 21 | 2,033 | 1,664 |
| MAG | 15 of 21 | 10, 8, 17 | 1,897 | 1,513 |
| WebQSP | 6 of 12 | 7, 2, 3 | 735 | 351 |
| MuSiQue | 13 of 19 | 12, 15, 11 | 1,649 | 1,265 |
| Amazon | 7 of 13 | 7, 11, 5 | 874 | 490 |

The **primary read is the unseen part**: every question of the split that is neither in the fit nor in the BRG5 screen reads (the screen questions were looked at, so they are kept apart). The remainder as a whole and the screen part are reported beside it.

## 3. Gates (no number is read unless all hold)

- **G-R0.** On every question of every chunk read, the `control` walk equals the engine's expansion and grade (as G-L0 of BRG5).
- **G-R1.** The chunks read are disjoint from the fit's chunks; the fit's training files are the six of Amendment 1 (three of BRG5, three of its `a1` folders) and the fitted positives per hop equal Amendment 1's `fit.json` entry for `lr6`.
- **G-R2.** On the three screen chunks the `lr6` row of every question (golds in the pool and pool size) equals Amendment 1's `lr6` row: 0 differences.
- **G-R3.** The chunks read are complete: the number of questions equals the split's questions minus 768.

## 4. What the read may say

Per set, on the unseen part, on the remainder and on the screen part: ALL and the share of golds in the pool for both variants, the mean pool size and the ratio to the control, the ceiling (golds among the candidates), the paired difference `lr6` minus `control` with a bootstrap of 2,000 resamples (seed 20261128) and the call at the 1.0-point line (`zdrp2.gain_call`). `reached` means ABOVE on the unseen part with a pool no larger than 1.02 times the control's. No claim of a rule: the read has one fit, five sets and a held-out part of the same split.

## 5. Priors (written before any number of this page)

- **N1.** Prime: `lr6` is ABOVE the control on the unseen part (lower end of the interval above +1.0), the point difference between +1.5 and +4.5.
- **N2.** MAG and WebQSP: the point differences on the unseen part are at least +0.5 and the calls are LEVEL (about 1,500 and 350 questions leave an interval of about plus or minus 1 and plus or minus 2).
- **N3.** MuSiQue and Amazon: the calls are LEVEL and the point differences within plus or minus 1.0.
- **N4.** Every set's mean pool for `lr6` is within 1.02 times the control's.
- **N5.** The control's ALL on the unseen part is within 2.0 points of the filed whole-split value of DRP1 (section 17; Prime 85.0, MAG 91.7, WebQSP 86.6, MuSiQue 86.2, Amazon 83.1) on every set.
- **N6.** G-R0 to G-R3 hold.

## 6. Rows

Five streams on the host, one per set (CPU only, two threads each), `zbrg6.py screen --ds D | --kb K --host`, then `read`; records `outputs/brg6/<set>/` (per-chunk `scr_c*.npz`, `gate_a6.json`) and `outputs/brg6/read_<set>.json`. The training matrices stay on the host (they are Amendment 1's). Module sha256 prefix `09e0996b6feb8e81` (local = host; committed 52d73c35; self-test PASS before any row).

## 7. Results (28 Nov 2026 21:16 IST)

Five streams on the host (CPU only, two threads each), 20:32 to 21:13 IST, one per set, module sha256 prefix `09e0996b6feb8e81`. The fit is Amendment 1's `lr6` (six chunks per set, 768 questions). The read covers every other chunk, and the questions read equal the split's minus 768 on all five sets (Prime 2,033, MAG 1,897, WebQSP 735, MuSiQue 1,649, Amazon 874). Each set's `gate_a6.json` has verdict PASS (G-R0 on every chunk, G-R1 disjoint chunks and the fitted positives of Amendment 1, G-R2 the three screen chunks' `lr6` rows equal to Amendment 1's, G-R3 complete). Records: `outputs/brg6/<set>/gate_a6.json`, `scr_c*.npz` and `outputs/brg6/read_<set>.json`.

Unseen part (the chunks outside the fit and outside the three screen chunks). Each cell: ALL and mean nodes of the control, then of `lr6` (ratio of its mean nodes to the control's), the ceiling (golds among the candidates), and the paired difference `lr6` minus `control` with its 95% interval and call.

| set | questions | control ALL @ nodes | `lr6` ALL @ nodes (ratio) | ceiling | difference [interval], call | reached |
| --- | --- | --- | --- | --- | --- | --- |
| Prime | 1,664 | 84.0 @ 2,071 | 87.3 @ 2,070 (1.000) | 96.2 | **+3.31 [+2.40, +4.33] ABOVE** | yes |
| MAG | 1,513 | 91.9 @ 2,044 | 93.2 @ 2,045 (1.001) | 97.9 | +1.32 [+0.40, +2.18] LEVEL | no |
| WebQSP | 351 | 87.2 @ 2,101 | 88.6 @ 2,105 (1.002) | 92.6 | +1.42 [-0.57, +3.42] LEVEL | no |
| MuSiQue | 1,265 | 90.4 @ 2,489 | 90.8 @ 2,489 (1.000) | 93.0 | +0.40 [-0.16, +0.95] LEVEL | no |
| Amazon | 490 | 82.7 @ 2,465 | 83.9 @ 2,465 (1.000) | 84.3 | +1.22 [+0.41, +2.25] LEVEL | no |

The remainder (the unseen part plus the three screen chunks) and the screen part alone give the same picture:

| set | remainder: questions, control / `lr6`, difference [interval], call | screen part: questions, control / `lr6`, difference [interval], call |
| --- | --- | --- |
| Prime | 2,033, 84.5 / 87.8, +3.25 [+2.41, +4.13] ABOVE | 369, 86.7 / 89.7, +2.98 [+1.08, +5.15] ABOVE |
| MAG | 1,897, 92.0 / 93.2, +1.21 [+0.47, +1.95] LEVEL | 384, 92.4 / 93.2, +0.78 [-0.52, +2.34] LEVEL |
| WebQSP | 735, 87.5 / 89.0, +1.50 [+0.14, +2.99] LEVEL | 384, 87.8 / 89.3, +1.56 [-0.52, +3.65] LEVEL |
| MuSiQue | 1,649, 89.4 / 89.7, +0.30 [-0.18, +0.85] LEVEL | 384, 86.2 / 86.2, +0.00 [-1.30, +1.30] LEVEL |
| Amazon | 874, 84.0 / 84.8, +0.80 [+0.34, +1.49] LEVEL | 384, 85.7 / 85.9, +0.26 [+0.00, +0.78] LEVEL |

Share of golds in the pool, unseen part, control to `lr6`: Prime 88.3 to 90.3, MAG 92.8 to 94.6, WebQSP 67.6 to 70.8, MuSiQue 95.7 to 96.0, Amazon 92.8 to 93.1.

Golds kept by hop, unseen part (kept golds over the golds that are candidates of that hop; the candidate sets differ a little between the two variants because hop 2 keeps different nodes): hop 2, Prime 1,314 of 1,654 to 1,398 of 1,653; MAG 1,043 of 1,257 to 1,117 of 1,253; WebQSP 599 of 942 to 661 of 947; MuSiQue 486 of 526 to 495 of 526; Amazon 732 of 746 to 741 of 746. Hop 3 (the sets with three hops), Prime 222 of 559 to 222 of 490; MAG 82 of 253 to 83 of 208; WebQSP 87 of 559 to 116 of 543. The gain sits at hop 2 on every set and, on WebQSP, also at hop 3.

**Calls on the priors.**

- **N1 HELD.** Prime: +3.31 [+2.40, +4.33] ABOVE, inside +1.5 to +4.5, on the unseen part (and ABOVE on the remainder and on the screen part).
- **N2 HELD.** MAG +1.32 and WebQSP +1.42, both at least +0.5, both LEVEL.
- **N3 NOT HELD.** MuSiQue +0.40 LEVEL is inside plus or minus 1.0; Amazon is LEVEL but its point difference, +1.22 [+0.41, +2.25], is outside plus or minus 1.0.
- **N4 HELD.** The mean pools are 1.000, 1.001, 1.002, 1.000, 1.000 times the control's (Prime, MAG, WebQSP, MuSiQue, Amazon).
- **N5 NOT HELD.** The unseen control against the filed whole-split value is Prime 84.0 against 85.0, MAG 91.9 against 91.7, WebQSP 87.2 against 86.6, Amazon 82.7 against 83.1, all within 2.0; MuSiQue is 90.4 against 86.2, 4.2 above. From the rounded values the fit chunks of MuSiQue read about 79 (86.2 on 2,417 questions against 89.4 on the 1,649 outside the fit). The chunks of a set are not exchangeable samples of it: the chunk order groups questions, so a read of the chunks outside the fit moves the level, and the paired difference is the quantity this page reads (a reading, not tested here).
- **N6 HELD.** G-R0 to G-R3, verdict PASS on all five sets.

**What it says.** A logistic regression on the 12 walk features, fitted once on six chunks, raises the share of questions with every gold in the pool on every set at the same pool size, by +0.4 (MuSiQue) to +3.3 (Prime) points on questions it never saw; only Prime's gain is ABOVE at the 1.0-point line (and it is ABOVE on all three parts), MAG's +1.32 has a lower end of +0.40, and WebQSP, MuSiQue and Amazon are LEVEL. Prime's 87.3 stays about 8 points below the target of 95: the order does not close any set. Nothing is adopted. A carry needs the same recipe on all ten datasets with a fit that does not use the read split's questions, and MetaQA's cap walk, where the same recipe is BELOW the arrival order (`docs/BRG7_LEARNED_ORDER_CAP_WALK.md` section 7), shows that it is not a universal order. Whether a fitted scorer belongs in the rules path or in the joint walk and rank model is the user's ruling.
