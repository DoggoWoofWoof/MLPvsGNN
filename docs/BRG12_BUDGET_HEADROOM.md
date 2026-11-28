# BRG12: the visited budget raised toward the 2,500-node line (screen)

Declared 28 Nov 2026 23:17 IST, before any number of this page. Screen on the first three chunks of WebQSP, Prime and MAG; nothing is adopted. Module: `outputs/mp_unified/zbrg12.py` (new; imports `zbrg9.py`, `zbrg1.py`, `zpl3g.py`, `zpl5g.py`, none edited; its self-test passed before this page was written; sha recorded in section 6).

## 1. Why

The target is 95 or more of the gold nodes in the pool at the smallest node count near 2,500. The cell of the dense-only arm on the three three-hop budget-following sets, `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (DRP1 section 17), visits at most 2,000 nodes and ends at 2,043 (MAG), 2,073 (Prime) and 2,099 (WebQSP) mean nodes on the whole split: 16 to 18 percent under the line. The filed best cell within 2,500 is chosen from a grid of budgets 2,000, 2,500, 5,000, 10,000 and 40,000, so the budgets between 2,000 and 2,500 were never walked; the 2,500 cells end above the line (2,500 plus the seeds and the rows that overshoot).

BRG1 (`docs/BRG1_BRIDGE_TRACE.md`) traced the missed golds of the three sets. Prime: 64.7 percent of the 85 missed golds are budget cuts, median 2.58 times the budget at the cutting hop (about 880 nodes there), 41 of 55 within 4 times. MAG: 80.6 percent of the 72 missed golds are budget cuts, median 2.25 times a hop budget of about 810. WebQSP: budget cuts are 37.5 percent of the missed golds (the window is the larger cause, 57.6 percent). A cut at 2 to 3 times the hop budget is far from what 400 more visited nodes buy, so the expected effect of using the headroom is small; this page measures how small. It is the only part of the walk's cost that the target allows and that has not been used: the order, the window, the depth, the beam, the row length and the hop split have been varied (BRG2 to BRG11), the budget has not.

## 2. What runs

`zbrg12.py screen --chunks 3` on WebQSP (384 questions), Prime (369) and MAG (384), the same three chunk positions as BRG5 and BRG11. Per question, six variants of one cell; nothing else changes (order key `fu`, window 512, families STRUCT, three hops, hop shares ceil(rem / (H - h + 1)), the first stage `dense_top200`):

| variant | visited budget |
| --- | --- |
| `eng` | the cell's own, 2,000 (the engine unpatched) |
| `same` | the wrapper with the cell's own budget (G-S0: equals `eng` on every question) |
| `v22` | 2,200 |
| `v23` | 2,300 |
| `v24` | 2,400 |
| `v25` | 2,500 (over the line by design: 2,500 plus the seeds' overshoot; read for the slope, not a candidate) |

MuSiQue and Amazon are not run: their cells (`rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`) end at 2,467 and 2,440 mean nodes with the first stage's 800, 1 and 2.5 percent under the line.

The wrapper replaces `zpl5g.expand5` for the duration of a variant's pass over a chunk and restores it after (also after an error); it calls the engine's function with `visited_cap` set to the variant's budget when the setting is budget-following, and with the setting unchanged otherwise.

## 3. Gates

- **G-C0.** The declared cell of each set is the filed best cell within 2,500 nodes (`outputs/drp1e/read_webqsp.json`, `outputs/drp1/read_<kb>.json`); `zbrg9.py` stops if not.
- **G-S0.** `same` equals `eng` per question (golds in the pool and pool size): 0 differences per chunk; the 128-question smoke prints this before the rows and nothing else of the smoke is used.
- **G-S1.** `eng` equals BRG5's control rows per question on the screen chunks: 0 differences.
- **G-S2.** The self-test of `zbrg12.py` passes before any row (BRG9's self-test; on 180 toy budget-following walks the wrapper with the cell's own budget equals `zpl5g.expand5` element for element, a raised budget keeps distinct fresh nodes inside the raised budget, 531 raised-budget runs differ from the engine's, the function is restored after a block and after an error, a non-budget-following setting is passed through): PASS, 28 Nov 2026.
- **G-S3.** The three chunks of every set are read.
- **G-S4** (for an amendment to the whole split, read time, no code). The whole-split `eng` ALL equals DRP1 section 17's filed value within 0.1 point and the question count is the filed one.

## 4. The read

`zbrg12.py read`: per set and variant ALL, the mean pool, the paired difference to `eng` in points with its interval and its call (ABOVE if the lower end of the 95 percent bootstrap interval is above +1.0, BELOW if the upper end is below -1.0, else LEVEL; 2,000 resamples, seed 20261128; `zdrp2.gain_call`). The `reached` flag of BRG9 does not apply (it asks for a pool of at most 1.05 times `eng`'s; here the pool is 1.17 to 1.22 times by design): the question is the ALL at a pool near 2,500, so the page also states each variant's mean pool in nodes. The 1.0-point rule needs a paired difference of about 2.5 points on 370 questions to call ABOVE, so a gain of 1 to 2 points reads LEVEL; the page therefore gives the point estimate, the interval and the number of questions that changed in each direction, and a gain of at least +1.0 on a set that its interval keeps above 0 is sent to the whole split (an amendment declared before its rows), as BRG11 did.

## 5. Priors (written before any row)

- **Z1.** G-S0 and G-S1 hold: 0 differences on every chunk of the three sets.
- **Z2.** `v24`'s mean pool is 2,430 to 2,530 on each of the three sets (the pool is the visited budget times the cell's ratio of 1.022 to 1.049, or the budget plus a constant of 43 to 99).
- **Z3.** `v25`'s ALL is at least `eng`'s on each set (point estimates), and `v24`'s point gain over `eng` is positive on each set and at most +3.0 points.
- **Z4.** No variant is ABOVE `eng` on any of the three sets.
- **Z5.** No variant reaches 95 on any set: WebQSP's best ALL stays under 92, Prime's under 90, MAG's under 95.
- **Z6.** The gain is not the same on the three sets: the largest of the three `v24` gains is at least twice the smallest (positive) one. (An expectation of a small, uneven effect; it fails if the three gains are alike.)

## 6. Rows

Three streams on the host, CPU only, two threads each: `zbrg12.py screen --ds webqsp | --kb prime | --kb mag --chunks 3 --host`. Module sha256 prefix `3cd07fb2e8ffb652` (host must equal local before the rows).

## 7. Results

None yet.
