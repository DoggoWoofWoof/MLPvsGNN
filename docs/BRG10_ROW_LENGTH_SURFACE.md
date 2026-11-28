# BRG10: the (L, H) surface of the row-length cap on the per-row-cap cells (whole split)

Declared 28 Nov 2026 22:08 IST, before any number of this page. Whole split of MetaQA, HotpotQA and SQuAD; nothing is adopted. Module: `outputs/mp_unified/zbrg10.py` (new; imports `zbrg9.py`, which is not edited and keeps the walk, the replacement cap functions, the gates G-S0 and G-S1 and the paired aggregate; its self-test runs first; sha recorded in section 6).

## 1. Why

BRG8 and BRG9 (`docs/BRG8_ROW_LENGTH_CAP.md`, `docs/BRG9_ROW_LENGTH_CAP_ALL.md`) found that a cap on the neighbours of a node that depends on the length of the row, with nothing fitted and no gold used, acts on the cells whose walk is capped by a small per-row constant: MetaQA's whole split goes from 93.3 at 2,485 nodes to 98.9 at 2,283 (rows of at most 50 entries whole, longer rows cut at 10), and on HotpotQA and SQuAD the same cut holds ALL within one point at pools 22 to 36 percent smaller on the screen. The seven pairs (L, H) of BRG9 are two pairs inherited from BRG8 and five more; they are a handful of points on a surface, and the best of them, V7 (50, 10), is the corner of the set. The user's target is 95 and above at the smallest pool (about 2,500 nodes or fewer), with one rule on all ten datasets. This page asks two things on the three per-row-cap sets whose whole split is cheap (MetaQA 3.5 seconds per chunk of 128 questions with nine variants): how small the pool can go while ALL holds, and whether one pair serves all three sets (the rule is one form, two constants).

A selection among many grid points on one read is not a rule, so the selection is made on one half of the chunks and checked on the other half (section 4), and the whole-split table is reported as a surface, not as a result of the chosen point.

## 2. What runs

**Sets and cells.** The cells of BRG9 section 2: MetaQA `rrf1000_b800+STRUCT:h3_c25_v2000` (cap 25), HotpotQA `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` and SQuAD `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (cap 100). e3v dense-only first stage, no K, no SPLADE.

**Instrument.** BRG9's: `zpl4g.capped_parts` replaced for the duration of one variant's pass by the length rule (a row of at most L entries kept whole, a longer row cut at H entries by the cell's own order), the engine's states loop and its grader `zpl3g.grade3`; `eng` is the engine unpatched and `same` the length-rule functions with the cell's own cap (G-S0). Variants are named `L<L>H<H>`.

**Grids.** MetaQA: L in {25, 40, 50, 75, 100, 150} and H in {3, 5, 8, 10, 15, 25} with H at most L, without (25, 25): 35 pairs. HotpotQA and SQuAD: L in {50, 100, 200, 400} and H in {3, 5, 10, 25, 50, 100} with H at most L, without (100, 100) (the cell itself): 22 pairs. BRG9's seven pairs are all grid points of the two 100-cap sets, and four of them, (50, 25), (100, 25), (100, 10) and (50, 10), are grid points of MetaQA's. Eight pairs are in all three grids: (50, 3), (50, 5), (50, 10), (50, 25), (100, 3), (100, 5), (100, 10), (100, 25).

**Rows.** `screen --ds D --chunks 0` on the whole split of each set (MetaQA 77 chunks, HotpotQA 58, SQuAD 93; 128 questions per chunk). BRG10 writes its own folder and runs every chunk, the first three included (G-S1 compares those with BRG9's rows).

## 3. Gates (no number is read unless all hold)

- **G-S0.** On every question `same` equals `eng`: 0 differences.
- **G-S1.** On the first three chunks of each set, `eng` and every grid point that is one of BRG9's seven variants equal BRG9's rows of the same questions (`outputs/brg9/<set>/scr_c*.npz`): 0 differences (this ties the new module to the filed one).
- **G-S2.** The self-test of `zbrg10.py` (BRG9's own self-test first; the grids; the selection rule on synthetic cases: the smallest eligible pool is picked, a smaller ineligible one is rejected, a point fails on the check half when its ALL falls, nothing is returned when nothing is eligible) passes before any row.
- **G-S3.** Every chunk of every set is read (the gate file gives PASS only when complete).
- **G-S4.** The whole-split question count and `eng`'s ALL equal DRP1 section 17's filed values within 0.15 point (MetaQA 9,785 questions and 93.3, HotpotQA 7,405 and 98.5, SQuAD 11,873 and 99.6); if not, no number of that set is used.

## 4. What the read may say

Per set: the table of every grid point on the whole split (ALL, mean pool and its ratio to `eng`'s, the paired difference to `eng` with a bootstrap of 2,000 resamples, seed 20261128, and the call at the 1.0-point line, `zdrp2.gain_call`), labelled a surface (descriptive; every point is read on the same questions).

**The declared choice.** Chunks are split by the parity of their position in the set's chunk order: `fit` is the even positions, `check` the odd ones. On `fit` a grid point is eligible when its ALL is at least max(0.95, `eng`'s ALL on `fit` minus 0.005); the choice is the eligible point with the smallest mean pool (ties: the smaller H, then the smaller L); none if none is eligible. On `check` the choice **holds** when its ALL is at least 0.95, its paired call against `eng` is not BELOW, and its pool ratio is at most its pool ratio on `fit` plus 0.03. The sentence that may be written for a set is "the rule picked (L, H) on `fit` and it holds (or does not hold) on `check`, ALL ... at ... nodes, ... of `eng`'s"; nothing else is called a result.

**Common pairs.** For each of the eight pairs in all three grids: eligible on `fit` of a set by the same threshold, and holding on `check` of the set by the same three clauses (ALL at least 0.95 and not BELOW; the pool clause is not applied). A pair is common when it is eligible and holding on all three sets. A common pair is a candidate for one rule on the per-row-cap cells, not an adoption: the five budget-following sets are BELOW under any pair of BRG9 (WebQSP, Amazon) or LEVEL (the rest), and no rule here touches them.

## 5. Priors (written before any number of this page)

- **U1.** G-S0, G-S1, G-S3 and G-S4 hold on all three sets.
- **U2 (MetaQA).** The rule picks a point on `fit` that holds on `check`, and its whole-split pool ratio is at most 0.90 (2,236 nodes or fewer against 2,485), with a whole-split ALL of at least 97.5.
- **U3 (HotpotQA).** The rule picks a point that holds on `check` with a whole-split pool ratio of at most 0.55 (1,037 nodes or fewer against 1,885).
- **U4 (SQuAD).** The same with a pool ratio of at most 0.55 (about 854 or fewer against 1,552).
- **U5.** On HotpotQA and SQuAD no grid point is ABOVE `eng` on the whole split (both are at the ceiling); on MetaQA at least ten of the 35 points are ABOVE.
- **U6.** At least one of the eight common pairs is eligible and holding on all three sets.
- **U7.** The choices of the three sets are not all the same pair (HotpotQA and SQuAD, which share a cell and a grid, pick the same pair or neighbours; MetaQA's pair has a smaller L).

## 6. Rows

Three streams on the host (CPU only, two threads each), `zbrg10.py screen --ds D --chunks 0 --host`, then `read` on the laptop after `rx fetch` of `outputs/brg10/<set>/` and the `common` table; records `outputs/brg10/<set>/` (per-chunk `scr_c*.npz`, `gate_s.json`) and `outputs/brg10/read_<set>.json`. Module sha256 prefix `f3c92fc99b03ef68` (self-test PASS before any row; local = host before the rows).

## 7. Results

None yet.
