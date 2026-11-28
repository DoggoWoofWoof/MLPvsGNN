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

## 7. Results

None yet.
