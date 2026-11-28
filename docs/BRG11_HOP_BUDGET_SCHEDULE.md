# BRG11: the split of the visited budget over the hops of a budget-following walk (screen)

Declared 28 Nov 2026 22:16 IST, before any number of this page. Screen on the first three chunks of the five budget-following sets; nothing is adopted. Module: `outputs/mp_unified/zbrg11.py` (new; imports `zbrg9.py`, `zbrg1.py`, `zpl3g.py`, `zpl4g.py`, `zpl5g.py`, none edited; its self-test runs before any row; sha recorded in section 6).

## 1. Why

BRG9 (`docs/BRG9_ROW_LENGTH_CAP_ALL.md` section 7) showed that the row-length rule belongs to the per-row-cap cells: it takes MetaQA to the target and is LEVEL or BELOW on the five budget-following sets, where the long rows are the ones that hold the golds. Those five are the sets that stay under 95 in the dense-only arm (MuSiQue 86.2, WebQSP 86.6, Prime 85.0, MAG 91.7, Amazon 83.1; DRP1 section 17), and BRG1 to BRG6 found what limits them: MuSiQue's golds mostly have no path within two hops (links), WebQSP's misses are a window and a few huge answer sets, and on Prime and MAG 65 and 81 percent of the misses are golds that are candidates and are cut by the budget at 2.3 to 2.6 times the budget. A perfect order at the cell's own frontier would give Prime +9.8, WebQSP +5.4, MAG +5.0, MuSiQue +3.1, Amazon +0.8, and no parameter-free key of the candidates (support, ppr, fused, depth, beam) beat the cell's own order (BRG2 to BRG4); a fitted order helps Prime only (BRG6).

One thing of the walk has not been varied: how the visited budget is split over the hops. The engine's walk (`zpl5g.expand5`) keeps at hop h at most ceil(rem / (H - h + 1)) fresh nodes, rem the budget left, so each hop takes an equal share of what is left and the last hop keeps the rest (a three-hop walk with a budget of 2,000 and 200 seeds: 600, then half of what is left, then the rest). The user named a per-hop limit on 26 Nov. That equal share is one point of a family; the golds of a set sit at some hop, and a split that matches where they sit spends the same budget on more of them. This page varies the split with nothing else changed: same candidate lists, same order key, same window, same visited budget, no gold and no label read by the rule.

## 2. What runs

**Cells.** Each set's filed best cell within 2,500 nodes of the dense-only arm (e3v, no K, no SPLADE), the same as in BRG9 section 2: MuSiQue `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`, WebQSP and STaRK-Prime and STaRK-MAG `dense_top200+STRUCT:h3_bf_v2000~cs-fu`, STaRK-Amazon `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`.

**The rule.** The share of hop h is ceil(rem * w_h / (w_h + ... + w_H)) with w_k = r^(k-1), in integers (r = num/den; the weights are num^(k-1) den^(H-k)), rem the visited budget minus the nodes visited so far; what a hop cannot fill passes to the next, and the last hop keeps what is left, as in the engine. r = 1 is the engine's own rule; r above 1 gives the later hops more, r below 1 the earlier ones. For a three-hop walk with 1,800 left after the seeds, hop 1 keeps 258 (r = 2), 86 (r = 4), 25 (r = 8) or 1,029 (r = 1/2) at the first hop; for a two-hop walk hop 1 keeps a third, a fifth, a ninth or two thirds of what is left.

**Variants.** `eng` (the engine unpatched), `same` (r = 1 through the copy; G-S0), `f2` (r = 1/2), `f4` (r = 1/4), `f8` (r = 1/8), `b2` (r = 2), `b4` (r = 4), `b8` (r = 8). Six schedules, three back-loaded and three front-loaded; none is fitted and none reads a label; which of them is best on a set is read from the screen, and a choice among them on one read is not a rule. `f4` and `f8` were added before any screen row, after the smoke of 128 questions per set (G-S0 only; its ALL values were printed, not recorded, not read) showed WebQSP's front-loaded variant above the engine's on those questions; the page's priors W1 to W7 were written before the smoke and are not changed, W8 is the prior of the added points.

**Instrument.** BRG9's: the engine's states loop (`zbrg1.states`) and its grader `zpl3g.grade3`, with `zpl5g.expand5` replaced for the duration of one variant's pass over a chunk and put back after (also after an error). Subset: the first three chunks of each set (128 questions each), the same as BRG9's screen and BRG5's control rows.

## 3. Gates (no number is read unless all hold)

- **G-C0.** The declared cell of each set is the filed best cell of the dense-only arm within 2,500 nodes (`outputs/drp1e/read_<ds>.json`, `outputs/drp1/read_<kb>.json`).
- **G-S0.** On every question `same` equals `eng` (golds in the pool and pool size): 0 differences.
- **G-S1.** On the three chunks `eng` equals BRG5's control rows of the same questions (golds in the pool and pool size): 0 differences.
- **G-S2.** The self-test of `zbrg11.py` passes before any row (BRG9's self-test; the integer shares equal the engine's at r = 1; on toy budget-following walks of two and three hops r = 1 equals `zpl5g.expand5` element for element, every schedule keeps distinct fresh nodes inside the visited budget, the schedules differ from the engine's; the function is restored after a block and after an error; a setting that is not budget-following is the engine's own).
- **G-S3.** The three chunks of every set are read.

## 4. What the read may say

Per set: ALL (every gold in the pool), the share of golds in the pool, the mean pool size and its ratio to `eng`'s, and the paired difference of each variant against `eng` with a bootstrap of 2,000 resamples (seed 20261128) and the call at the 1.0-point line (`zdrp2.gain_call`). Across the five sets: for each variant the number of sets ABOVE, LEVEL and BELOW, and the pool ratios. A variant ABOVE on a set is a candidate for one whole-split read on that set (an amendment declared first); nothing is adopted from a screen, and no schedule is called a rule unless it is at least LEVEL on all five sets.

## 5. Priors (written before any number of this page)

- **W1.** G-S0 and G-S1 hold on all five sets.
- **W2.** Every variant's mean pool is within 0.97 to 1.03 of `eng`'s on every set (the same visited budget; a hop that cannot fill passes the rest on).
- **W3.** On Prime and MAG, at least one of `b2`, `b4`, `b8` is ABOVE `eng` on at least one of the two sets (the budget cut sits at the last hop).
- **W4.** On WebQSP, MuSiQue and Amazon no variant is ABOVE (WebQSP's misses are the window and a few huge answer sets, MuSiQue's the links, Amazon's the window and the lexical gap).
- **W5.** `f2` is not ABOVE on any of the five sets.
- **W6.** `b8` is BELOW `eng` on at least one set (the first hop keeps too little).
- **W7.** No variant is at least LEVEL on all five sets and ABOVE on two or more.
- **W8.** (Written after the smoke, before any screen row.) On WebQSP at least one of `f2`, `f4`, `f8` is ABOVE `eng`, and none of the front-loaded variants is ABOVE on Prime, MAG, MuSiQue or Amazon.

## 6. Rows

Five streams on the host (CPU only, two threads each), `zbrg11.py screen --ds D | --kb K --host`, then `read` on the laptop after `rx fetch` of `outputs/brg11/<set>/`; records `outputs/brg11/<set>/` (per-chunk `scr_c*.npz`, `gate_s.json`) and `outputs/brg11/read_<set>.json`. Module sha256 prefix `b2544915449ea838` (self-test PASS before any row; local = host before the rows).

## 7. Results of the screen (28 Nov 2026 23:10 IST)

`zbrg11.py screen --chunks 3` on the five sets, module sha256 prefix `b2544915449ea838` (local = host; committed 27ea83fe and 2e332674; self-test PASS; the smokes of 128 questions per set were run on the first version of the module, G-S0 0 differences on all five), five streams on the host 22:37 to 23:09 (two threads each, with BRG10's two streams running beside them). All gates PASS on every set: G-C0, G-S0 (`same` equals `eng` on every question of the three chunks: 0 differences on all five), G-S1 (`eng` equals BRG5's control rows per question on all five: 0 differences), G-S3. Records: `outputs/brg11/<set>/gate_s.json`, `scr_c*.npz` and `outputs/brg11/read_<set>.json`.

Per set, first cell: ALL and mean nodes of `eng` (the cell of DRP1 section 17 as the engine gives it on these questions). Then the six schedules: ALL, and in brackets the paired difference to `eng` in points and its call (A above, L level, B below at the 1.0-point line, 2,000 bootstrap resamples, seed 20261128). f2, f4, f8 give the earlier hops more (r = 1/2, 1/4, 1/8), b2, b4, b8 the later ones (r = 2, 4, 8).

| set | questions | `eng` | f2 | f4 | f8 | b2 | b4 | b8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue | 384 | 86.2 @ 2,467 | 86.2 (+0.00 L) | 86.2 (+0.00 L) | 86.2 (+0.00 L) | 86.5 (+0.26 L) | 86.2 (+0.00 L) | 85.2 (-1.04 L) |
| WebQSP | 384 | 87.8 @ 2,097 | 89.3 (+1.56 L) | 88.8 (+1.04 L) | 89.1 (+1.30 L) | 82.8 (-4.95 B) | 77.3 (-10.42 B) | 64.3 (-23.44 B) |
| Prime | 369 | 86.7 @ 2,076 | 86.4 (-0.27 L) | 87.5 (+0.81 L) | 87.3 (+0.54 L) | 86.4 (-0.27 L) | 83.5 (-3.25 L) | 79.9 (-6.78 B) |
| MAG | 384 | 92.4 @ 2,045 | 92.4 (+0.00 L) | 92.7 (+0.26 L) | 93.5 (+1.04 L) | 92.2 (-0.26 L) | 89.6 (-2.86 L) | 85.9 (-6.51 B) |
| Amazon | 384 | 85.7 @ 2,440 | 85.7 (+0.00 L) | 85.7 (+0.00 L) | 85.7 (+0.00 L) | 85.7 (+0.00 L) | 85.7 (+0.00 L) | 85.4 (-0.26 L) |

Mean pool of each variant over `eng`'s:

| set | f2 | f4 | f8 | b2 | b4 | b8 |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue | 1.000 | 1.000 | 1.000 | 1.000 | 1.003 | 1.009 |
| WebQSP | 0.999 | 0.999 | 0.999 | 1.001 | 1.005 | 1.009 |
| Prime | 0.999 | 0.998 | 0.998 | 1.002 | 1.005 | 1.011 |
| MAG | 0.999 | 0.998 | 0.998 | 1.002 | 1.006 | 1.012 |
| Amazon | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 |

Over the five sets, the calls against `eng` per variant: f2 0 above, 5 level, 0 below; f4 0 above, 5 level, 0 below; f8 0 above, 5 level, 0 below; b2 0 above, 4 level, 1 below; b4 0 above, 4 level, 1 below; b8 0 above, 2 level, 3 below. Sets where a variant is BELOW: f2: none; f4: none; f8: none; b2: WebQSP; b4: WebQSP; b8: WebQSP, Prime, MAG. WebQSP's intervals of the front-loaded variants: f2 [+0.26, +3.12], f4 [-0.53, +2.86], f8 [-0.52, +3.12] (points).

**Calls on the priors.**

- **W1 HELD.** G-S0 and G-S1: 0 differences on all five sets.
- **W2 HELD.** Every variant's mean pool is 0.998 to 1.012 of `eng`'s (the largest, b8 on MAG, +1.2 percent).
- **W3 NOT HELD.** No back-loaded variant is ABOVE on Prime or MAG (b2 is -0.27 on Prime and -0.26 on MAG; b4 -3.25 and -2.86; b8 -6.78 and -6.51, BELOW).
- **W4 HELD.** WebQSP, MuSiQue and Amazon: no variant is ABOVE (WebQSP's best, f2, is +1.56 LEVEL; MuSiQue's and Amazon's variants are within +0.26 to -1.04 with pools equal).
- **W5 HELD.** `f2` is LEVEL on all five sets (WebQSP +1.56, Prime -0.27, MuSiQue, MAG and Amazon +0.00).
- **W6 HELD.** `b8` is BELOW on WebQSP (-23.44), Prime (-6.78) and MAG (-6.51).
- **W7 HELD.** No variant is ABOVE on any set.
- **W8 NOT HELD.** The first clause fails: none of f2, f4, f8 is ABOVE on WebQSP (+1.56, +1.04, +1.30, all LEVEL; the lower end of f2's interval is above 0 and under 1.0). The second clause holds: none of them is ABOVE on Prime, MAG, MuSiQue or Amazon.

**What it says.** The split of the budget over the hops matters on the three-hop walks only: on the two-hop walks (MuSiQue, Amazon) ALL and the pool hardly move under any schedule (+0.26 to -1.04, pools within 0.9 percent; the first hop appears to fill less than its share, read from the pool sizes, not traced). On the three-hop sets back-loading starves the first hop and is BELOW on WebQSP under all three (-4.95, -10.42, -23.44) and under b8 on Prime and MAG. Front-loading is never BELOW and its point estimates are positive on WebQSP (+1.0 to +1.6) and MAG (0.0 to +1.0, f8) and mixed on Prime (-0.3 to +0.8): small effects that 369 to 384 questions cannot separate from zero by the 1.0-point rule. The engine's equal share is therefore within about one point of the best of this family; at most a front-loaded share adds about one point on WebQSP and MAG. No schedule is called a rule, nothing is adopted; one whole-split read of the front-loaded variants on the three-hop sets decides whether the one point is real (Amendment 1, section 8). None of this is near 95 (WebQSP 86.6, Prime 85.0, MAG 91.7 at the whole split): the budget cut is not mainly a question of which hop gets the budget.

## 8. Amendment 1 (declared 28 Nov 2026 23:11 IST, before any row of it): the whole split of WebQSP, Prime and MAG

Why. The screen finds no variant ABOVE on any set (section 7). The front-loaded schedules (f2, f4, f8) are LEVEL on all five sets and never BELOW, their pools equal the engine's to within 0.2 percent, and on the three-hop walks their point estimates are positive on WebQSP (+1.04 to +1.56; f2's interval [+0.26, +3.13] lies above 0 and not above 1.0) and MAG (+0.26 and +1.04 for f4 and f8) and mixed on Prime (-0.27 to +0.81). A small effect of that size is not separated from zero by 369 to 384 questions; it is read on the whole split of the three sets where the split matters (the two-hop walks do not move): WebQSP (1,503 questions, 12 chunks), Prime (2,801, 22 chunks) and MAG (2,665, 21 chunks). No code change: `screen --chunks 0` with the same module (sha `b2544915449ea838`), which keeps the three chunk files of the screen and adds the others, then `read`; the back-loaded variants are computed with the rest (the module has one variant list) and are read as the screen's priors say.

Gate G-S4 (read time, no code): the whole-split `eng` ALL of a set equals DRP1 section 17's filed value within 0.1 point (WebQSP 86.6, Prime 85.0, MAG 91.7) and the question count is the filed one (1,503, 2,801, 2,665); if not, no number of that set is used. G-S0 holds per chunk as before, and G-S1 on the three screen chunks.

Read. As in BRG9 section 8: the whole split, the unseen part (every chunk outside the three screen chunks) and its two halves (even and odd chunk positions) per set, the same calls and bootstrap; a variant is `reached` on a set when it is ABOVE on the unseen part and on both halves with a mean pool of at most 1.05 times `eng`'s.

Priors (written before any row of this amendment).

- **X1.** G-S0 and G-S4 hold on the three sets.
- **X2.** WebQSP: at least one of f2, f4, f8 is ABOVE `eng` on the unseen part (the screen's +1.0 to +1.6 holds at the whole split).
- **X3.** Prime and MAG: no variant is ABOVE on the unseen part.
- **X4.** No variant brings ALL to 95 or above on any of the three sets (WebQSP's best stays under 90, Prime's under 88, MAG's under 94).
- **X5.** b2, b4 and b8 are BELOW `eng` on WebQSP on the whole split; b8 is BELOW on Prime and on MAG.
- **X6.** Every variant's mean pool is within 0.98 and 1.02 of `eng`'s on all three sets.

Rows: three streams on the host (CPU only, two threads each), `zbrg11.py screen --kb prime|mag | --ds webqsp --chunks 0 --host`.
