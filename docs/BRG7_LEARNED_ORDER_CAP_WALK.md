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

## 7. Results

None yet.
