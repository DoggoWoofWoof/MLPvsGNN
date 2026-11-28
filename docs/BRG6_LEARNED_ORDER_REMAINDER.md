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

Five streams on the host, one per set (CPU only, two threads each), `zbrg6.py screen --ds D | --kb K --host`, then `read`; records `outputs/brg6/<set>/` (per-chunk `scr_c*.npz`, `gate_a6.json`) and `outputs/brg6/read_<set>.json`. The training matrices stay on the host (they are Amendment 1's). Module sha: recorded here after the self-test.

## 7. Results

None yet.
