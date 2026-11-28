# BRG9: the row-length cap on all ten datasets (screen)

Declared 28 Nov 2026 21:23 IST, before any number of this page. Screen on the first three chunks of every set; nothing is adopted. Module: `outputs/mp_unified/zbrg9.py` (new; imports `zbrg1.py`, `zbrg4.py`, `zdrp1.py`, `zdrp2.py` and the pool engine's `zpl2g.py`, `zpl3g.py`, `zpl4g.py`, `zpl5g.py`, none edited; its self-test runs before any row; sha recorded in section 6).

## 1. Why

BRG8 (`docs/BRG8_ROW_LENGTH_CAP.md` section 7, filed f2e9c841) found on MetaQA that a cap on the neighbours of a node that depends on the length of the row, with nothing fitted and no gold used, takes the whole split from 93.3 at 2,485 nodes to 98.2 at 2,501 (rows of at most 50 entries kept whole, longer rows cut at 25) and 98.9 at 2,413 (at most 100 whole, longer cut at 10). The user's target is one rule on all ten datasets (95 and above at about 2,500 nodes). Nine datasets are unread under this rule: four reach the target already (2Wiki 96.7, HotpotQA 98.5, SQuAD 99.6, MultiHop-RAG 98.8 in DRP1 section 17) and five do not (MuSiQue 86.2, WebQSP 86.6, STaRK-Prime 85.0, STaRK-MAG 91.7, STaRK-Amazon 83.1). BRG1 to BRG4 put the cause of the five at links (MuSiQue), the budget (Prime, MAG), the window and the budget (WebQSP) and the window and the first stage (Amazon); a cap that cuts the hub rows hard and keeps the middle rows whole acts on the budget and the window, so it is worth reading on every set before any new link or model work. This page reads it on all ten with one set of constants, and says which sets it helps, leaves alone or hurts.

## 2. What runs

**Cells.** Each set's filed best cell within 2,500 nodes of the dense-only arm (no K, no SPLADE, e3v), the same as in BRG1, BRG4 and DRP1 section 17: 2Wiki `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq`, HotpotQA `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr`, MetaQA `rrf1000_b800+STRUCT:h3_c25_v2000`, MuSiQue `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`, SQuAD `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800`, WebQSP `dense_top200+STRUCT:h3_bf_v2000~cs-fu`, MultiHop-RAG `rrf1000_b50+FULL:h2_c100_v5000+top50`, STaRK-Prime and STaRK-MAG `dense_top200+STRUCT:h3_bf_v2000~cs-fu`, STaRK-Amazon `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`.

**Rule.** For every node the walk expands and every family it reads, the row of the node in the family keeps all its entries when the row has at most `L` entries and the first `H` in the cell's own order (the best `H` by cosine in the cells with `cs` and in the budget-following cells) otherwise. The cell's own cap (25 or 100 entries per row) or window (512, the budget-following cells) is the `eng` row below. Nothing else of the walk changes: the order of the candidates, the visited budget, the hops, the seeds, the first stage and the pool are the engine's.

**Variants.** `eng`: the engine's own pool of the cell. `same`: the length-rule functions with the cell's own cap (a check: it must equal `eng` on every question). `V1` (`L` 50, `H` 25), `V2` (100, 25), `V3` (100, 10), `V4` (200, 10), `V5` (200, 25), `V6` (100, 50), `V7` (50, 10). V1, V2 and V3 are BRG8's `L50`, `L100` and `L100h10`.

**Instrument.** The two per-row cap functions of the pool engine (`zpl4g.capped_parts` for the per-node-cap cells, `zpl5g.capped_parts5` for the budget-following cells) are replaced by the length-rule versions for the duration of one variant's pass over a chunk and restored after (also on an error); the engine's own states loop and grade (`zpl3g.grade3`) give every variant's golds in the pool and pool size. No walk is reimplemented, so the cells with `top`, `s20`, `cs` and the budget-following walks need no separate gate.

**Subset.** The first three chunks (positions 0 to 2, up to 128 questions each, as BRG5's screen chunks) of each set's chunk order: DRP1's `chunk_order` for the seven, STK3's for the STaRK sets. About 384 questions per set, so the interval on a paired difference is about plus or minus 2 points on a set with a high ALL and wider on a hard one. The whole split is a second stage, declared by an amendment on this page after the screen is read, for the sets and variants the screen leaves open (and for all ten for the variants the screen leaves universal), before its rows.

## 3. Gates (no number is read unless all hold)

- **G-C0.** The declared cell of each set is the filed best cell of the dense-only arm within 2,500 nodes (`outputs/drp1e/read_<ds>.json`, `outputs/drp1/read_<kb>.json`).
- **G-S0.** On every question `same` equals `eng` (golds in the pool and pool size): 0 differences.
- **G-S1.** On the screen chunks `eng` equals the earlier filed rows of the same questions where there are any (BRG5's control rows for MuSiQue, WebQSP, Prime, MAG and Amazon; BRG8's `base` for MetaQA), and V1, V2 and V3 on MetaQA equal BRG8's `L50`, `L100` and `L100h10`: 0 differences. The four sets with no earlier rows (2Wiki, HotpotQA, SQuAD, MultiHop-RAG) are checked by G-S0 alone on the screen and against the filed whole-split ALL in the second stage.
- **G-S2.** The self-test of `zbrg9.py` (the two replacement functions equal the engine's when given the cell's own cap or window, with and without `cs`, on toy graphs; the functions are restored after a with-block and after an error) passes before any row.
- **G-S3.** The three chunks of every set are read.

## 4. What the read may say

Per set on the screen part: ALL (every gold in the pool), the share of golds in the pool, the mean pool size and its ratio to `eng`'s, and the paired difference of each variant against `eng` with a bootstrap of 2,000 resamples (seed 20261128) and the call at the 1.0-point line (`zdrp2.gain_call`). Then across the ten sets: for each variant the number of sets ABOVE, LEVEL and BELOW and the pool ratios; the variants that are at least LEVEL (not BELOW) on all ten are the candidates for a universal rule, and the page says which sets keep 95 and above. A variant picked among seven on one read of 384 questions has a selection bias and a wide interval: nothing is a rule at this stage. Nothing is adopted.

## 5. Priors (written before any number of this page)

- **S1.** G-S0, G-S1 and G-S3 hold (V1, V2, V3 on MetaQA equal BRG8's rows per question).
- **S2.** On each of 2Wiki, HotpotQA, SQuAD and MultiHop-RAG, V3 is not BELOW `eng` on the screen part (LEVEL or ABOVE) with a mean pool no larger than `eng`'s, on at least three of the four.
- **S3.** On Prime or on MAG at least one of V1 to V7 is ABOVE `eng` on the screen part (a hard cut of the hub rows frees the visited budget the two sets run out of).
- **S4.** On WebQSP and Amazon no variant is ABOVE `eng` (their loss is the window and the first stage, not the flood).
- **S5.** On MuSiQue no variant is ABOVE `eng` (links are the lever there).
- **S6.** No single variant is at least LEVEL (not BELOW) on all ten sets: the constants that fit MetaQA's rows do not fit the rows of every graph.

## 6. Rows

Nine streams on the host (CPU only, two threads each), one per set, `zbrg9.py screen --ds D | --kb K --host`; MultiHop-RAG on the laptop (its inputs are laptop-only), then `read` on the laptop; records `outputs/brg9/<set>/` (per-chunk `scr_c*.npz`, `gate_s.json`) and `outputs/brg9/read_<set>.json`. Module sha256 prefix `ee33c22782d00628` (self-test PASS before any row).

## 7. Results of the screen (28 Nov 2026 22:01 IST)

`zbrg9.py screen --chunks 3` on every set, module sha256 prefix `ee33c22782d00628` (local = host; committed 6956bbf1; self-test PASS; smokes of 16 questions on WebQSP, 2Wiki and MultiHop-RAG before the rows), nine streams on the host and MultiHop-RAG on the laptop, 21:25 to 22:00. All gates PASS on every set: G-C0 (the declared cell is the filed best), G-S0 (`same` equals `eng` on every question of every chunk read: 0 differences on all ten sets), G-S1 (on the screen chunks `eng` equals BRG5's control rows for MuSiQue, WebQSP, Prime, MAG and Amazon and BRG8's `base` for MetaQA, and MetaQA's V1, V2, V3 equal BRG8's `L50`, `L100`, `L100h10` per question: 0 differences), G-S3 (the three chunks of each set). Records: `outputs/brg9/<set>/gate_s.json`, `scr_c*.npz` and `outputs/brg9/read_<set>.json`.

Screen part, per set. First cell: ALL and mean nodes of `eng` (the cell of DRP1 section 17, as the engine gives it on these questions). Then V1 to V7: ALL, and in brackets the paired difference to `eng` in points and its call (A above, L level, B below at the 1.0-point line, 2,000 bootstrap resamples, seed 20261128).

| set | questions | `eng` | V1 (50, 25) | V2 (100, 25) | V3 (100, 10) | V4 (200, 10) | V5 (200, 25) | V6 (100, 50) | V7 (50, 10) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2Wiki | 384 | 96.4 @ 2,225 | 96.6 (+0.26 L) | 97.1 (+0.78 L) | 97.1 (+0.78 L) | 97.1 (+0.78 L) | 97.1 (+0.78 L) | 97.1 (+0.78 L) | 96.6 (+0.26 L) |
| HotpotQA | 384 | 97.9 @ 1,885 | 97.9 (+0.00 L) | 97.9 (+0.00 L) | 97.7 (-0.26 L) | 97.9 (+0.00 L) | 97.9 (+0.00 L) | 97.9 (+0.00 L) | 97.1 (-0.78 L) |
| MetaQA | 384 | 93.8 @ 2,509 | 98.2 (+4.43 A) | 97.1 (+3.39 A) | 99.2 (+5.47 A) | 94.8 (+1.04 L) | 93.5 (-0.26 L) | 92.2 (-1.56 L) | 99.5 (+5.73 A) |
| MuSiQue | 384 | 86.2 @ 2,467 | 83.6 (-2.60 L) | 83.1 (-3.12 B) | 82.6 (-3.65 B) | 83.3 (-2.86 L) | 83.9 (-2.34 L) | 84.9 (-1.30 L) | 80.7 (-5.47 B) |
| SQuAD | 384 | 99.7 @ 1,552 | 99.2 (-0.52 L) | 99.7 (+0.00 L) | 99.7 (+0.00 L) | 99.7 (+0.00 L) | 99.7 (+0.00 L) | 99.7 (+0.00 L) | 99.0 (-0.78 L) |
| WebQSP | 384 | 87.8 @ 2,097 | 69.8 (-17.97 B) | 75.8 (-11.98 B) | 68.2 (-19.53 B) | 75.0 (-12.76 B) | 80.5 (-7.29 B) | 77.1 (-10.68 B) | 61.5 (-26.30 B) |
| MultiHop-RAG | 384 | 98.2 @ 100 | 98.2 (+0.00 L) | 98.2 (+0.00 L) | 98.2 (+0.00 L) | 98.2 (+0.00 L) | 98.2 (+0.00 L) | 98.2 (+0.00 L) | 97.9 (-0.26 L) |
| Prime | 369 | 86.7 @ 2,076 | 82.9 (-3.79 L) | 84.0 (-2.71 L) | 79.4 (-7.32 B) | 84.0 (-2.71 L) | 86.4 (-0.27 L) | 86.4 (-0.27 L) | 77.2 (-9.49 B) |
| MAG | 384 | 92.4 @ 2,045 | 92.4 (+0.00 L) | 92.7 (+0.26 L) | 92.7 (+0.26 L) | 93.0 (+0.52 L) | 92.7 (+0.26 L) | 91.7 (-0.78 L) | 92.4 (+0.00 L) |
| Amazon | 384 | 85.7 @ 2,440 | 81.8 (-3.91 B) | 81.8 (-3.91 B) | 81.5 (-4.17 B) | 82.8 (-2.86 B) | 82.8 (-2.86 B) | 83.1 (-2.60 B) | 80.5 (-5.21 B) |

Mean pool of each variant over `eng`'s:

| set | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2Wiki | 1.000 | 1.000 | 1.001 | 1.001 | 1.000 | 1.000 | 1.001 |
| HotpotQA | 0.698 | 0.750 | 0.707 | 0.804 | 0.839 | 0.830 | 0.642 |
| MetaQA | 1.005 | 1.017 | 0.966 | 1.027 | 1.043 | 1.047 | 0.923 |
| MuSiQue | 0.894 | 0.978 | 0.981 | 0.990 | 0.984 | 0.980 | 0.806 |
| SQuAD | 0.698 | 0.784 | 0.754 | 0.959 | 0.968 | 0.862 | 0.670 |
| WebQSP | 0.842 | 0.928 | 0.774 | 0.936 | 0.984 | 0.987 | 0.520 |
| MultiHop-RAG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.977 |
| Prime | 0.940 | 0.965 | 0.907 | 0.983 | 0.989 | 0.990 | 0.751 |
| MAG | 1.002 | 1.002 | 1.004 | 1.004 | 1.001 | 1.000 | 1.004 |
| Amazon | 0.663 | 0.733 | 0.711 | 0.769 | 0.783 | 0.774 | 0.612 |

Over the ten sets, the calls against `eng` per variant: V1 1 above, 7 level, 2 below; V2 1 above, 6 level, 3 below; V3 1 above, 5 level, 4 below; V4 0 above, 8 level, 2 below; V5 0 above, 8 level, 2 below; V6 0 above, 8 level, 2 below; V7 1 above, 5 level, 4 below. Sets where a variant is BELOW: V1: WebQSP, Amazon; V2: MuSiQue, WebQSP, Amazon; V3: MuSiQue, WebQSP, Prime, Amazon; V4: WebQSP, Amazon; V5: WebQSP, Amazon; V6: WebQSP, Amazon; V7: MuSiQue, WebQSP, Prime, Amazon.

**Calls on the priors.**

- **S1 HELD.** G-S0 (0 differences, ten sets), G-S1 (0 differences, six sets with earlier rows; MetaQA's V1, V2, V3 equal BRG8's per question), G-S3.
- **S2 HELD.** V3 is not BELOW `eng` with a mean pool no larger than `eng`'s on HotpotQA (97.7 against 97.9, nodes 1,334 against 1,885), SQuAD (99.7 against 99.7, 1,170 against 1,552) and MultiHop-RAG (98.2 against 98.2, 100 against 100): three of four. 2Wiki's V3 is LEVEL (+0.78) with a pool of 2,227 against 2,225, 0.1 percent larger.
- **S3 NOT HELD.** No variant is ABOVE on Prime (the best, V5 and V6, are -0.27, LEVEL) or on MAG (the best, V4, is +0.52, LEVEL).
- **S4 HELD.** WebQSP: all seven variants are BELOW (-7.29 to -26.30); Amazon: all seven BELOW (-2.60 to -5.21).
- **S5 HELD.** MuSiQue: no variant is ABOVE (V1, V4, V5, V6 are LEVEL at -2.60, -2.86, -2.34, -1.30; V2, V3, V7 are BELOW at -3.12, -3.65, -5.47).
- **S6 HELD.** No variant is at least LEVEL on all ten sets: every one is BELOW on WebQSP and on Amazon.

**What it says.** The row-length cap acts on the cells whose walk is capped by a small per-row constant. On MetaQA it is ABOVE under V1, V2, V3 and V7 (+4.4 to +5.7 on the screen; the whole split is read in section 8 and comes out as BRG8's). On HotpotQA and SQuAD it leaves ALL where it is (not BELOW, within 0.8 point) at pools 22 to 36 percent smaller under V1, V2, V3 and V7 (V1 on HotpotQA 0.70 of `eng`'s nodes, on SQuAD 0.70; V3 0.71 and 0.75; V7 0.64 and 0.67). It does nothing on 2Wiki (the pool sits at the visited cap of 2,000 plus the head: +0.26 to +0.78 LEVEL), MultiHop-RAG (a pool of 100 nodes) and MAG (-0.78 to +0.52 LEVEL, pools equal). It hurts the budget-following cells, where the long rows are the ones that hold the golds: WebQSP -7 to -26, Amazon -2.6 to -5.2, Prime BELOW under V3 and V7 and LEVEL (-0.27 to -3.79) under the rest, MuSiQue -1.3 to -5.5. So one pair of constants is not universal on the ten as a cell-independent rule (S6), and the rule belongs to the per-row-cap family of cells, as the per-set cell choice of DRP1 already was. Each number is the best of seven variants read on 384 questions; the whole split is the read that counts, and Amendment 1 (section 8) reads it on the four cap-cell sets named there (MetaQA and MultiHop-RAG read, HotpotQA and SQuAD running). 2Wiki, the one cap-cell set with no stage 2, is left at the screen: its seven variants are all LEVEL with an unchanged pool, so a whole-split read (about 600 seconds per chunk on the contended host) would not change a call. Nothing is adopted.

## 8. Amendment 1 (declared 28 Nov 2026 21:45 IST, before any row of it): the whole split of the four finished cap-cell sets

Why. Seven of the ten screens are read (section 7 is written when all ten are in): MetaQA is ABOVE `eng` under V1, V2, V3 and V7 (V7 99.5 at 2,315 nodes against 93.8 at 2,509 on the screen); HotpotQA and SQuAD keep their ALL under V1 to V6 with pools 25 to 30 percent smaller (HotpotQA V1 97.9 at 1,315 against 97.9 at 1,885; SQuAD V3 99.7 at 1,170 against 99.7 at 1,552); MultiHop-RAG does not move (pool of 100 nodes); the budget-following cells are LEVEL or BELOW under every variant (MuSiQue, Amazon, Prime). The cap cells are where the rule acts, and the target reads on the whole split, so the whole split of MetaQA, HotpotQA, SQuAD and MultiHop-RAG is read now with the same module and the same nine variants (no code change: `screen --chunks 0`, which keeps the three chunk files of the screen and adds the others). 2Wiki, WebQSP, MAG (screens still running) and the budget-following sets get their own amendment after their screens are read.

Gate G-S4 (read time, no code): the whole-split `eng` ALL of a set equals DRP1 section 17's filed value to within 0.1 point (MetaQA 93.3, HotpotQA 98.5, SQuAD 99.6, MultiHop-RAG 98.8); if it does not, no number of that set is used. G-S0 holds per chunk as before, and G-S1 holds on the three screen chunks. For MetaQA the whole-split V1, V2 and V3 are compared with BRG8's whole-split `L50`, `L100` and `L100h10` (98.2, 97.6, 98.9) as a check of the engine route.

Read. The page reads the whole split, the unseen part (every chunk outside the three screen chunks) and its two halves (even and odd chunk positions) per set, with the same calls and bootstrap as section 4. A variant is `reached` on a set when it is ABOVE on the unseen part and on both halves with a mean pool no larger than 1.05 times `eng`'s; on a set that is already at 95 or above the effect that matters is the pool, so the page also gives, per variant, the pool ratio and whether ALL stays within 1.0 point (not BELOW) of `eng`.

Priors (written before any row of this amendment).

- **T1.** G-S4 holds on all four sets and MetaQA's V1, V2, V3 equal BRG8's whole-split values within 0.1 point.
- **T2.** MetaQA: V7 is ABOVE `eng` on the unseen part and on both halves with a pool ratio of at most 0.97, and its whole-split ALL is at least 98.5.
- **T3.** HotpotQA and SQuAD: at least one of V1 to V7 is not BELOW `eng` on the unseen part with a pool ratio of at most 0.85, on each of the two sets.
- **T4.** MultiHop-RAG: every variant is within 0.5 point of `eng` and within 0.02 of its pool.
- **T5.** No variant is ABOVE on HotpotQA, SQuAD or MultiHop-RAG (they are at the ceiling already).

Rows: four streams (MetaQA, HotpotQA, SQuAD on the host, CPU only, two threads each; MultiHop-RAG on the laptop), `zbrg9.py screen --ds D --chunks 0`, then `read`; module sha unchanged.
