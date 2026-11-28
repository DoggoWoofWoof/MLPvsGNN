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

## 7. Results

None yet.

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
