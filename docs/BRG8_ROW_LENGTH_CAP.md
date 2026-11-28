# BRG8: a cap that depends on the length of the row (MetaQA's cap walk)

Declared 28 Nov 2026 21:13 IST, before any number of this page. Read on the whole split of MetaQA; nothing is adopted. Module: `outputs/mp_unified/zbrg8.py` (new; imports `zbrg4.py` and `zbrg7.py`, neither edited; its self-test runs before any row; sha recorded in section 6).

## 1. Why

BRG7 (`docs/BRG7_LEARNED_ORDER_CAP_WALK.md`, section 7, filed 35ed58e7) read the learned candidate order on MetaQA's cap walk (8,633 unseen questions): BELOW the arrival order at caps 25 (-4.29) and 50 (-5.39 against the same cap) and ABOVE it only at cap 100 (+5.17, still 7.5 below the cell). The hop-2 candidates are all kept, so only the last hop's visited cap binds, and there the arrival order keeps 97.1 percent of the golds that are candidates. MetaQA's misses are not in the order. BRG4 found them in the row cap: golds beyond the 25th neighbour of a frontier node's row (UPSTREAM 61.8, WINDOW 20.6 percent of the misses), and a uniform larger cap floods the visited cap (cap 50 93.0 LEVEL, cap 100 81.8 BELOW, with ceilings of 100.0 and 99.7).

The rows that make a uniform cap flood are the long ones (a genre, a year, a language, a tag lists hundreds to thousands of films), and the rows that lose golds are likely the middle ones (an actor or a director lists a few dozen films, and the question asks for them). A cap that depends on the length of the row, with no fitted parameter, keeps a middle row whole and cuts a long row at the cell's 25 (or less): the ceiling of the larger cap without its flood. This page measures that.

## 2. What runs

MetaQA, the cell of BRG4 (`rrf1000_b800+STRUCT:h3_c25_v2000`), the walk of `zbrg4.trace_cap_walk` (arrival order, visited cap 2,000, no cosine) with the per-row keep rule changed: for every frontier node and family, a row of at most `L` entries is kept whole, a longer row is cut at `H` entries. Six variants (`L`, `H`): `base` (25, 25: the cell itself), `L50` (50, 25), `L100` (100, 25), `L200` (200, 25), `L100h10` (100, 10), `L200h10` (200, 10). The seeds' hop uses the same rule. Nothing is fitted: no training chunks.

Chunks of 128 questions in the set's chunk order (as BRG4). The screen part is the three chunks of BRG4 (positions 0 to 2, seen); the read covers every chunk (77), and the primary read is the **unseen part** (every chunk except the three, 8,633 questions), split by chunk position into two halves (even and odd positions) for the stability check of section 4.

## 3. Gates (no number is read unless all hold)

- **G-N0.** On every question of every chunk `base` equals the engine's expansion and grade.
- **G-N1.** On the screen chunks `base` reproduces BRG4's `cap25_control` row of every question (golds in the pool, pool size): 0 differences.
- **G-N2.** The walk with the rule at `L` = 25, `H` = 25 and the walk of `zbrg7.cap_walk_screen` at cap 25 give equal expansions on the toy graphs of the self-test (the self-test runs before any row).
- **G-N3.** All 77 chunks are read: the question count equals the split's.

## 4. What the read may say

On the unseen part, its two halves, the screen part and the whole split: ALL, the share of golds in the pool, the mean pool size and its ratio to `base`'s, the ceiling (golds among the candidates), and the paired difference of every variant against `base` with a bootstrap of 2,000 resamples (seed 20261128) and the call at the 1.0-point line. `reached` means ABOVE on the unseen part and on each half, with a pool no larger than 1.05 times `base`'s. A variant picked among six on one read has a selection bias: a carry would need the same variant ABOVE on both halves, which is the check built into `reached`. The target is read off the whole-split ALL of the best variant beside `base`'s 93.3 (95 and above at about 2,500 nodes). Nothing is adopted.

## 5. Priors (written before any number of this page)

- **R1.** At least one of `L100`, `L200`, `L100h10`, `L200h10` is ABOVE `base` on the unseen part, the best point difference between +0.5 and +3.0.
- **R2.** `L50` is LEVEL (between -0.5 and +1.0): the uniform cap 50 was LEVEL and most rows lost are longer than 50.
- **R3.** The ceiling of `L100` is at least 98.0 (BRG4: 99.7 at the uniform cap 100, 95.3 at cap 25), and its pool is within 1.05 times `base`'s.
- **R4.** `L100h10` is above `L100` in ALL (the room freed by the hub rows is used).
- **R5.** G-N0 to G-N3 hold.

## 6. Rows

One stream on the host (CPU only, two threads), `zbrg8.py screen --ds metaqa --host`, then `read` on the laptop; records `outputs/brg8/metaqa/` (per-chunk `scr_c*.npz`, `gate_n.json`) and `outputs/brg8/read_metaqa.json`. Module sha256 prefix `6a0b4a8bb051e91e` (self-test PASS on 60 toy walks before any row; the declared G-N2 is the self-test's equality of the `L = H = 25` walk with `zbrg4` and `zbrg7`).

## 7. Results (28 Nov 2026 21:18 IST)

`screen --ds metaqa` on the host, module sha256 prefix `6a0b4a8bb051e91e` (local = host; committed c174dd22; self-test PASS on 60 toy walks; smoke of 24 questions before the row), one stream, 21:13 to 21:16 (77 chunks, about 2 s each). All gates PASS: G-N0 0 mismatches between `base` and the engine on all 77 chunks, G-N1 0 differences against BRG4's `cap25_control` on the three screen chunks (384 questions), G-N2 in the self-test, G-N3 9,785 questions read of 9,785. Records: `outputs/brg8/metaqa/gate_n.json`, `scr_c*.npz` and `outputs/brg8/read_metaqa.json`.

**Correction to the declaration text (no rule changed).** Section 2 gives the unseen part as 8,633 questions, which is BRG7's figure (its fit took six more chunks). BRG8 fits nothing, so the unseen part is the 74 chunks outside the three screen chunks: 9,401 questions (69,796 golds); the screen part is 384 questions, the whole split 9,785.

Unseen part (9,401 questions). Each cell: ALL; share of golds in the pool; mean nodes (ratio to `base`); ceiling (golds among the candidates); difference to `base` [95% interval], call; `reached` (ABOVE on the unseen part and on both halves, pool at most 1.05 times `base`'s).

| variant (`L`, `H`) | ALL | golds in pool | mean nodes (over `base`) | ceiling | minus `base` [interval], call | reached |
| --- | --- | --- | --- | --- | --- | --- |
| `base` (25, 25) | 93.3 | 93.5 | 2,484 (1.000) | 94.1 | | |
| `L50` (50, 25) | 98.1 | 97.0 | 2,500 (1.007) | 99.2 | **+4.85 [+4.39, +5.30] ABOVE** | yes |
| `L100` (100, 25) | 97.6 | 95.9 | 2,541 (1.023) | 99.9 | **+4.29 [+3.78, +4.81] ABOVE** | yes |
| `L200` (200, 25) | 92.4 | 87.7 | 2,617 (1.054) | 99.9 | -0.89 [-1.54, -0.29] LEVEL | no |
| `L100h10` (100, 10) | **98.9** | 98.3 | 2,413 (0.971) | 99.9 | **+5.58 [+5.08, +6.08] ABOVE** | yes |
| `L200h10` (200, 10) | 95.1 | 91.8 | 2,573 (1.036) | 99.9 | +1.79 [+1.20, +2.35] ABOVE | no (half odd +0.84 LEVEL; pool 1.036) |

The two halves (even and odd chunk positions; 4,665 and 4,736 questions) agree for `L50` (+5.10 and +4.60), `L100` (+4.74 and +3.84) and `L100h10` (+5.96 and +5.22), all ABOVE; `L200` is +0.32 LEVEL and -2.09 BELOW, `L200h10` +2.74 ABOVE and +0.84 LEVEL. The screen part (384 questions) gives `base` 93.8, `L50` 98.2, `L100` 97.1, `L200` 93.5, `L100h10` 99.2, `L200h10` 94.8, and the whole split (9,785 questions): `base` 93.3 @ 2,485, `L50` 98.2 @ 2,501 (+4.83 [+4.38, +5.27] ABOVE), `L100` 97.6 @ 2,541 (+4.25), `L200` 92.4 @ 2,618 (-0.87 LEVEL), **`L100h10` 98.9 @ 2,413 (+5.58 [+5.08, +6.06] ABOVE)**, `L200h10` 95.1 @ 2,573 (+1.76).

Golds that are candidates of the last hop and the number the walk keeps within the visited cap, unseen part: `base` 35,356 of 36,414 (97.1 percent); `L50` 36,780 of 38,114 (96.5); `L100` 35,339 of 38,340 (92.2); `L200` 29,310 of 38,338 (76.5); `L100h10` 37,134 of 38,382 (96.7); `L200h10` 32,352 of 38,378 (84.3). Hops 1 and 2 keep every gold that is a candidate in every variant (hop 2 candidates: 22,034 for `base`, 23,184 for `L50`, 23,810 for `L100`, 23,766 for `L100h10`).

**Calls on the priors.**

- **R1 NOT HELD.** The best of the four larger variants is `L100h10`, ABOVE as the prior said, but the point difference is +5.58, outside +0.5 to +3.0.
- **R2 NOT HELD.** `L50` is +4.85 [+4.39, +5.30], ABOVE (the prior said LEVEL, between -0.5 and +1.0).
- **R3 HELD.** The ceiling of `L100` is 99.9 (at least 98.0) and its pool is 1.023 times `base`'s.
- **R4 HELD.** `L100h10` 98.9 is above `L100` 97.6.
- **R5 HELD.** G-N0 to G-N3.

**What it says.** On MetaQA the row cap is the lever, as BRG4 found, and a cap that depends on the length of the row is the first parameter-free-in-the-labels change of the walk that clears the target on this set: the whole split goes from 93.3 at 2,485 nodes to 98.2 at 2,501 (`L50`) and 98.9 at 2,413 (`L100h10`), with no fitted parameter and no gold used. The ceiling rises from 94.2 to 99.3 or 99.9 and the walk keeps 96.5 to 96.7 percent of the golds that are candidates, so the visited cap is still the loss that is left (the share kept falls to 76.5 percent when long rows are cut only at 25 and `L` is 200: the flood of the long rows is what `H` = 10 removes). The rule has two constants chosen among six variants on one read: three of the six are ABOVE on the unseen part and on both halves with a pool within 1.05, and the best of them is the one with the smallest cut of the long rows, so the selection does not explain the size of the effect, but the constants (50 or 100 against the cell's cap of 25, and 25 or 10) are not yet a rule. Whether a row-length cap carries to the other nine datasets, where the cells have their own caps (25 on 2Wiki, 100 on HotpotQA and MultiHop-RAG), is the question of the next page: BRG9 (`docs/BRG9_ROW_LENGTH_CAP_ALL.md`). Nothing is adopted.
