# BRG7: the learned candidate order on MetaQA's cap walk (caps 25, 50, 100)

Declared 28 Nov 2026 20:35 IST, before any number of this page. Screen on three chunks and a read on every chunk of the split outside the fit; nothing is adopted. Module: `outputs/mp_unified/zbrg7.py` (new; imports `zbrg4.py`, `zbrg5.py`, `zbrg5b.py`, none edited; its self-test runs before any row; sha recorded in section 6).

## 1. Why

MetaQA is the one set below the target whose misses are not links (BRG1: no path 0 percent) and not a budget over a cosine-scored window: its cell is a per-node cap of the first 25 neighbours with a visited cap of 2,000 (`rrf1000_b800+STRUCT:h3_c25_v2000`, whole-split dense-only ALL 93.3). BRG4 traced it: about 82 percent of the missed golds are cut by the 25-neighbour row cap (UPSTREAM 61.8, WINDOW 20.6), and the ceiling (golds among the candidates) is 95.3 at cap 25, 100.0 at cap 50 and 99.7 at cap 100; but a larger cap with the cell's own order (arrival) floods the visited cap and falls (cap 50 93.0, LEVEL; cap 100 81.8, BELOW), and no parameter-free order (support, ppr, cosine) beats arrival. The headroom of a perfect order at cap 50 is +7.0.

BRG5 and its Amendment 1 found that a per-hop logistic regression on 12 walk features, fitted on other chunks of the same split, orders a hop's candidates better than the cell's own order wherever a budget binds (Prime ABOVE at an unchanged pool, +3.0 to +4.6; MAG and WebQSP LEVEL). MetaQA's visited cap is such a budget from hop 2 on. This page runs the same instrument on the cap walk, at caps 25, 50 and 100, to see whether a learned order lets a larger cap use its higher ceiling.

It is a diagnostic, not a model of record and not one of the two families (MLP, GNN), as in BRG5. A positive read is a result on one set; a rule needs the recipe unchanged on all ten datasets and the user's ruling on a fitted scorer in the rules path.

## 2. What runs

MetaQA, the cell above, the walk of `zbrg4.trace_cap_walk` (per-node cap on every hop, visited cap kept, no cosine inside the cap), with the cap set to 25, 50 or 100 on both the seeds' hop and the frontiers' hops. Six variants: `c25`, `c50`, `c100` (the cell's own order, arrival) and `l25`, `l50`, `l100` (the same walk with the distinct fresh candidates of hops 2 and 3 reordered by the score of a logistic regression, standardized, C = 1, 200 iterations, per cap and per hop, on the 12 features of BRG5, computed from that cap's candidate lists). A hop with fewer than 20 positives keeps the arrival order. The features are BRG5's `hop_features`, unchanged; hop 1 (every neighbour of the seeds) has no budget and is not scored.

Chunks of 128 questions in the set's chunk order (as BRG4). The fit: chunks at positions 3 to 8 (768 questions; negatives subsampled per chunk as in BRG5, at most 133,333 per hop and chunk). The screen: positions 0 to 2 (BRG4's three chunks). The read: every other chunk, positions 9 and up (the unseen part, primary). All three variants of the fit are fitted once, before any evaluation chunk is walked.

## 3. Gates (no number is read unless all hold)

- **G-M0.** On every question the `c25` walk equals the engine's expansion and grade (as G-C1 of BRG4).
- **G-M1.** Training chunks disjoint from the read chunks; per cap and hop the number of positives in the collected rows equals an independent recount of the golds among the candidate lists; no non-finite feature.
- **G-M2.** On the three screen chunks `c25`, `c50` and `c100` reproduce BRG4's `cap25_control`, `cap50_control` and `cap100_control` rows of every question (golds in the pool, pool size): 0 differences.
- **G-M3.** The read covers every chunk outside positions 3 to 8, and the question count equals the split's minus the fit's.

## 4. What the read may say

On the unseen part, the screen part and the remainder: ALL, the share of golds in the pool, the mean pool size and its ratio to `c25`'s, the ceiling, and the paired differences with a bootstrap of 2,000 resamples (seed 20261128) and the call at the 1.0-point line: every variant against `c25` (the cell), and `l25`, `l50`, `l100` against `c25`, `c50`, `c100` of the same cap. `reached` means ABOVE `c25` on the unseen part with a pool no larger than 1.05 times `c25`'s (BRG4's controls at caps 50 and 100 already have pools of 1.045 and 1.070 times `c25`'s: 2,623 and 2,685 against 2,509 nodes, because the first hop keeps more neighbours of the seeds); the pool ratio is reported beside every number. The target (95 and above on the whole split, at about 2,500 nodes) is read off `l50`'s or `l25`'s unseen ALL, with the whole-split control (93.3) beside it; nothing is adopted and no claim of a rule.

## 5. Priors (written before any number of this page)

- **P1.** `l25` against `c25` is LEVEL, the point difference between -0.5 and +1.5: at cap 25 the cut is the row cap, which an order inside the visited cap does not touch.
- **P2.** `l50` against `c25` is ABOVE on the unseen part (lower end above +1.0): the ceiling at cap 50 is 100.0 and the control order wastes it.
- **P3.** `l100` against `c100` is ABOVE with a point difference of at least +3.0, and `l100` against `c25` is BELOW (upper end below -1.0): the flood at cap 100 is too large for the order to undo.
- **P4.** `l50` is above `l25` and `l100` in ALL.
- **P5.** Every learned variant's pool is within 1.02 times the pool of the cell's own order at the same cap (BRG4: 2,509, 2,623 and 2,685 nodes).
- **P6.** The unseen `c25` ALL is within 2.0 points of the filed whole-split value 93.3.
- **P7.** G-M0 to G-M3 hold.

Correction of section 4 and prior P5 (before any row of this page, after reading BRG4's pool sizes for the caps): the first wording compared every pool with `c25`'s within 1.02, which the cap-50 and cap-100 controls themselves exceed by construction (1.045 and 1.070); the wording above compares within the same cap, and `reached` allows 1.05 times `c25`'s.

## 6. Rows

One stream on the host (CPU only, two threads), `zbrg7.py screen --ds metaqa --host`, then `read` on the laptop; records `outputs/brg7/metaqa/` (per-chunk `scr_c*.npz`, `gate_m.json`, `fit.json`) and `outputs/brg7/read_metaqa.json`. The training matrices stay on the host. Module sha: recorded here after the self-test.

## 7. Results (28 Nov 2026 21:11 IST)

`screen --ds metaqa` on the host, module sha256 prefix `c00428636a681398` (local = host; committed 14fdd2d8; self-test PASS on 288 walks; smoke of 24 questions before the row), one stream, 20:38 to 21:07. The fit: chunks 70, 4, 43 are the screen chunks (positions 0 to 2); the fit chunks are positions 3 to 8; 71 chunks read (8,633 unseen questions with 64,369 golds, 384 screen questions). G-M0 PASS (0 mismatches between the `c25` walk and the engine on all 71 chunks), G-M1 PASS (recount 0, non-finite 0 on the six training chunks), G-M2 PASS (`c25`, `c50`, `c100` reproduce BRG4's three controls on the three screen chunks), G-M3 PASS (complete). Positives per hop in the fit (cap 25 / 50 / 100): hop 2 1,749 / 1,898 / 1,901, hop 3 2,566 / 2,720 / 2,727 (hop 1 is not scored). Records: `outputs/brg7/metaqa/gate_m.json`, `fit.json`, `scr_c*.npz` and `outputs/brg7/read_metaqa.json`.

Unseen part (8,633 questions). Each cell: ALL; mean nodes (ratio to `c25`); ceiling (golds among the candidates); difference to `c25` [95% interval], call; difference to the control of the same cap, call.

| variant | ALL | mean nodes (over `c25`) | ceiling | minus `c25` [interval], call | minus same-cap control, call |
| --- | --- | --- | --- | --- | --- |
| `c25` (the cell) | 93.3 | 2,478 (1.000) | 94.2 | | |
| `l25` | 89.0 | 2,470 (0.997) | 94.2 | -4.29 [-4.75, -3.83] BELOW | -4.29, BELOW |
| `c50` | 92.8 | 2,616 (1.056) | 99.5 | -0.56 [-1.20, +0.06] LEVEL | |
| `l50` | 87.4 | 2,606 (1.052) | 99.5 | -5.94 [-6.64, -5.26] BELOW | -5.39 [-6.03, -4.70] BELOW |
| `c100` | 80.6 | 2,689 (1.085) | 99.9 | -12.70 [-13.59, -11.87] BELOW | |
| `l100` | 85.8 | 2,676 (1.080) | 99.9 | -7.53 [-8.29, -6.79] BELOW | **+5.17 [+4.54, +5.83] ABOVE** |

The screen part (384 questions) reproduces BRG4's numbers (`c25` 93.8, `c50` 93.0, `c100` 81.8) and the remainder (9,017 questions) the unseen part's (`c25` 93.3, `l25` 89.0, `c50` 92.8, `l50` 87.4, `c100` 80.7, `l100` 85.8).

Golds that are candidates of hop 3 (the last hop, where the visited cap binds) and the share that the walk keeps within the budget, unseen part: `c25` 32,858 of 33,848 (97.1 percent); `l25` 29,367 (86.8); `c50` 27,694 of 35,360 (78.3); `l50` 24,938 (70.5); `c100` 14,538 of 35,312 (41.2); `l100` 19,900 (56.4). Hop 2 never binds: every gold among the hop-2 candidates is kept by every variant (20,285 of 20,285 at cap 25; 21,908 at cap 50; 22,073 at cap 100).

**Calls on the priors.**

- **P1 NOT HELD.** `l25` against `c25` is -4.29 [-4.75, -3.83], BELOW (the prior said LEVEL, between -0.5 and +1.5).
- **P2 NOT HELD.** `l50` against `c25` is -5.94 [-6.64, -5.26], BELOW (the prior said ABOVE).
- **P3 HELD.** `l100` against `c100` is +5.17 [+4.54, +5.83] (at least +3.0), and `l100` against `c25` is -7.53 [-8.29, -6.79], upper end below -1.0.
- **P4 NOT HELD.** `l50` (87.4) is above `l100` (85.8) and below `l25` (89.0).
- **P5 HELD.** The pools of the learned variants are 0.997, 0.996 and 0.995 times their same-cap controls'.
- **P6 HELD.** The unseen `c25` ALL is 93.3, the filed whole-split value.
- **P7 HELD.** G-M0 to G-M3.

**What it says.** On MetaQA's cap walk a logistic regression on the 12 walk features ranks the last hop's candidates worse than their arrival order at the point that matters, the cut at the remaining visited cap: it keeps 86.8 percent of the golds that are candidates against 97.1 at cap 25, and 70.5 against 78.3 at cap 50. It is ABOVE only where the arrival order floods (cap 100, +5.17), and that is still 7.5 points below the cell. The arrival order is not an arbitrary one here: candidates arrive grouped under their frontier node in the row's own order, and the golds of a question arrive as blocks under a few parents, which a score of each candidate on its own cannot keep (a reading, not tested on this page). MetaQA's misses are not an order problem, as BRG4 found for the parameter-free orders; they are the row cap, and the screen leaves the row as the thing to change: BRG8 (`docs/BRG8_ROW_LENGTH_CAP.md`).
