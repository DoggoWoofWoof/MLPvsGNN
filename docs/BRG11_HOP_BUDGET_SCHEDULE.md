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

## 7. Results

None yet.
