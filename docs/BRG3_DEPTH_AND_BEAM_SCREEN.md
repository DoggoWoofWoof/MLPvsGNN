# BRG3: do more hops, with a narrow frontier, reach the golds beyond the cell's hops?

Declared 28 Nov 2026 13:28 IST, after BRG1 (`docs/BRG1_BRIDGE_TRACE.md`, section 6) and before any number of this page. BRG2's reads of MuSiQue, WebQSP, Amazon and Prime were open when this page was written (MAG's was not); they are named where a prior uses them. Read-only screen on the subsets of BRG1; nothing is adopted and no cell or rule of the filed work changes.

## 1. Why

BRG1 found that on MuSiQue 71.9 percent of the missed golds have no path within the cell's two hops of the five seeds, on Amazon 35.1 percent, on Prime 14.1 percent (plus 14.1 percent upstream cuts). CR1 (filed) found the chain's golds within two links of each other for 84.2 percent of MuSiQue's slots, at a median of three intermediate nodes, so a third or fourth gold of a chain is three to five links from the seeds. The pool grids of PL2 to PL6 walked one to three hops; four hops and more were never walked (UL2H's ledger: distance 4 = beyond three). BRG2 showed that the order inside the budget is no lever for MuSiQue (the ceiling of a perfect order at the walk's own frontier is 89.3 against 86.2): what is missing there is reach.

Depth is expensive in a budget-following walk because each hop's budget is the remaining budget over the remaining hops, and the frontier of the next hop is the whole kept set. A **beam** (the next hop reads the rows of only the first nodes of the kept set, in the question's order) makes depth cheap: the "per-hop limit guided by the question" of the user's list. This page tests both.

## 2. What runs

`outputs/mp_unified/zbrg3.py` (new; imports `zbrg1.py` and `zbrg2.py` and the pool code, edits none). The walk of the declared cell (order, window 512, visited cap 2,000, the cell's families) is re-run hop by hop as in BRG1's `trace_walk` with the hop count and the frontier of the later hops replaced:

| variant | hops | frontier of hops 2 and later |
| --- | --- | --- |
| `control` | the cell's own (MuSiQue and Amazon 2, Prime 3) | the whole kept set (the engine's walk) |
| `h3` | 3 | the whole kept set |
| `h4` | 4 | the whole kept set |
| `h3b100` | 3 | the first 100 kept nodes in the cell's order |
| `h4b100` | 4 | the first 100 kept nodes in the cell's order |
| `h5b100` | 5 | the first 100 kept nodes in the cell's order |

The hop budget is the cell's rule over the new hop count (remaining visited budget over the remaining hops), the candidate order is the cell's, the head and the seeds are the cell's. On Prime `h3` equals `control` by construction (a check, not a variant). The beam size 100 is declared here and not tuned on these reads.

Per question and variant: the golds in the pool (head and expansion), the pool size, and the **ceiling** (golds in the head or among any hop's candidates: the pool a perfect order would give at this variant's own frontier).

**Sets, cells, subset.** MuSiQue, Amazon and Prime (the sets whose misses are 14 percent or more beyond the hops; WebQSP's and MAG's are under 2 percent and 0 percent), at BRG1's declared cells and its three chunks of the seeded order. **Gates.** G-D0 = BRG1's G-B0, G-D1, G-E0 (imported). **G-H1**: `control` reproduces the engine's expansion and its grade on every question of every chunk (a mismatch stops the chunk, no number of that set is read). The self-test: PASS before this page was committed (beam 0 equals `expand5` on 80 toy walks over four settings; deeper and beamed walks keep distinct fresh nodes inside the visited cap).

**The read** (`zbrg3.py read`): per variant ALL, golds in the pool, mean nodes, the ceiling, ALL by gold count, and against `control` the paired difference with a 2,000-resample bootstrap over questions (seed 20261128) and the call of `zdrp2.gain_call` (ABOVE if the lower end is above +1.0 point, BELOW if the upper end is below -1.0, else LEVEL). Subset reads are screens.

## 3. Priors (written before any number of this page)

- **D1 (MuSiQue, depth alone).** `h3` and `h4` are not ABOVE `control` (the budget per hop shrinks as the hops grow; the filed grids did not prefer three hops over two at this visited cap).
- **D2 (MuSiQue, beam).** At least one of `h3b100`, `h4b100`, `h5b100` is ABOVE `control` on MuSiQue (a narrow frontier buys the chain's third and fourth gold without spending the budget on every kept node's row), and that variant's ceiling is at least 3 points above `control`'s ceiling.
- **D3 (Amazon).** No variant is ABOVE `control` on Amazon (its misses are window cuts at hop 2 and hub rows; its 800-deep head holds 89 percent of the golds).
- **D4 (Prime).** No variant is ABOVE `control` on Prime (its misses are budget cuts; more hops spread the same budget).
- **D5 (size).** Every variant's mean pool size is within 5 percent of `control`'s on every set.
- **D6 (G-H1).** `control` equals the engine on every question of every chunk (0 mismatches).

## 4. What the read may say

- Each prior is judged against every value of its sentence.
- A variant is a candidate for one later declaration (a whole-split read on all ten datasets under one rule) only if it is ABOVE on at least two of the three sets and BELOW on none. If no variant is ABOVE anywhere, depth and a question-ordered beam are closed as levers for this walk in this form (the rule, the cells and the beam of 100) and MuSiQue's remaining distance is in the links.
- Nothing from SPLADE, K, name seeds or BM25 enters (the dense-only arm's K store is empty, so the FULL cells walk S and R). The hub line stays closed (8.15).

## 5. Rows

CPU only, on the host through `rx exec` streams (`pylib_run.py outputs/mp_unified/zbrg3.py depth --ds D | --kb K --host`), one stream per set. A `read` per set on the laptop after the fetch.

## 6. Results (28 Nov 2026 13:40 IST)

**What ran.** `depth` on MuSiQue, Amazon and Prime, three chunks each (BRG1's; 384, 384 and 369 questions), module sha256 prefix `73cf9a930ff01701`, three streams on the host. G-H1 PASS on every chunk (0 mismatches). `control` equals BRG1's and BRG2's `control@512` on all three sets (86.2, 85.7, 86.7), and Prime's `h3` equals `control` as constructed. Records: `outputs/brg3/<set>/gate_depth.json`, `dep_c*.npz`, `read_<set>.json`. Subset reads (screens).

**Table 1. ALL (percent) by variant, the paired difference to `control` in points with its interval and the call; the ceiling (a perfect order at the variant's own frontier) and the mean pool size over the control's.**

| set | variant | ALL | minus `control` [95% interval], call | ceiling | pool size over `control` |
| --- | --- | --- | --- | --- | --- |
| MuSiQue (cell h2) | `control` | 86.2 | | 89.3 | 1.000 |
| | `h3` | 85.2 | -1.04 [-3.12, +1.04] LEVEL | 94.0 | 0.905 |
| | `h4` | 83.6 | -2.60 [-5.21, -0.25] LEVEL | 94.0 | 0.887 |
| | `h3b100` | 85.7 | -0.52 [-2.60, +1.56] LEVEL | 87.5 | 0.981 |
| | `h4b100` | 85.2 | -1.04 [-3.39, +1.04] LEVEL | 88.3 | 0.958 |
| | `h5b100` | 83.9 | -2.34 [-4.95, +0.00] LEVEL | 89.1 | 0.942 |
| Amazon (cell h2) | `control` | 85.7 | | 86.5 | 1.000 |
| | `h3` | 86.7 | +1.04 [-0.26, +2.60] LEVEL | 88.0 | 0.976 |
| | `h4` | 87.0 | +1.30 [+0.00, +2.86] LEVEL | 88.3 | 0.942 |
| | `h3b100` | 85.2 | -0.52 [-2.08, +0.78] LEVEL | 86.5 | 0.972 |
| | `h4b100` | 85.7 | +0.00 [-1.56, +1.56] LEVEL | 87.0 | 0.993 |
| | `h5b100` | 85.9 | +0.26 [-1.30, +1.82] LEVEL | 87.2 | 0.982 |
| Prime (cell h3) | `control` (= `h3`) | 86.7 | | 96.5 | 1.000 |
| | `h4` | 85.9 | -0.81 [-2.44, +0.54] LEVEL | 97.3 | 0.988 |
| | `h3b100` | 88.9 | +2.17 [+0.27, +4.07] LEVEL | 93.8 | 1.011 |
| | `h4b100` | 88.1 | +1.36 [-0.27, +3.25] LEVEL | 95.1 | 1.002 |
| | `h5b100` | 87.8 | +1.08 [-0.54, +2.71] LEVEL | 95.4 | 0.997 |

By gold count (4 or more golds): MuSiQue 72.7 (`control`), 69.5 (`h3`), 64.1 (`h4`), 69.5, 69.5, 68.0 (the three beams); Amazon 72.7, 74.5, 75.2, 71.5, 72.7, 73.3; Prime 79.2, 79.2 (`h3`), 75.5, 83.0, 77.4, 75.5.

**Calls on the priors.**

- **D1 HELD.** MuSiQue `h3` -1.04 and `h4` -2.60 are LEVEL (the upper end of `h4` is -0.25, above the -1.0 line), neither ABOVE.
- **D2 NOT HELD.** No beam variant is ABOVE on MuSiQue (-0.52, -1.04, -2.34), and their ceilings (87.5, 88.3, 89.1) are below `control`'s 89.3, not 3 points above.
- **D3 HELD.** Amazon: none ABOVE; `h4` is the nearest (+1.30 [+0.00, +2.86]).
- **D4 HELD by the declared call, with a direction to report.** Prime: none ABOVE (the interval of `h3b100`, [+0.27, +4.07], lies above 0 and not above the +1.0 line); the three beams are +2.17, +1.36 and +1.08 and `h4` -0.81.
- **D5 NOT HELD.** Pool sizes within 5 percent of `control`'s fail on MuSiQue (`h3` 0.905, `h4` 0.887, `h5b100` 0.942) and Amazon (`h4` 0.942); Prime's stay within 1.2 percent. More hops spend less of the visited budget when the candidates run short.
- **D6 HELD.** 0 mismatches on all 9 chunks.

**What the screen says.**

1. **Depth and a question-ordered beam do not reach MuSiQue's missed golds: no variant is ABOVE on any of the three sets**, so by the page's section 4 rule (ABOVE on at least two of three, BELOW on none) depth and a beam of 100 are **closed as levers for this walk in this form**. MuSiQue's remaining distance is not the hop count.
2. **The golds beyond the cell's hops are among the candidates once the walk goes deeper, and the order does not keep them.** MuSiQue's ceiling is 94.0 at three and four hops (against 89.3 at two) while its ALL falls to 85.2 and 83.6: the golds are among the candidates of the later hops and the order that keeps them does not pick them (BRG2: the question's cosine, which found the chain's first golds, beats the frontier-agreement orders, and none of the orders tried recovers the ceiling). On Amazon the ceiling at four hops is 88.3 (+1.8) and the ALL +1.30.
3. **Prime leans the other way for a narrow frontier**: all three beams are positive (+1.08 to +2.17, ALL 87.8 to 88.9) while the same variants give -0.52 to -2.34 on MuSiQue and -0.52 to +0.26 on Amazon; with one set positive, one level and one negative it is not a lever, and it is the one direction of this page worth testing on the other two sets whose cells walk three hops (WebQSP, MAG): Amendment 1 below.

Nothing is adopted.

## 7. Amendment 1 (28 Nov 2026 13:41 IST): the beam on the other two three-hop cells

Declared after section 6 and before any number of WebQSP or MAG on this page. No code change: `zbrg3.py` (sha prefix `73cf9a930ff01701`) runs `--ds webqsp` and `--kb mag` at their declared cells (`dense_top200+STRUCT:h3_bf_v2000~cs-fu`), the same three chunks of BRG1, the same six variants (`h3` equals `control` there by construction), the same beam of 100 and the same gates. Reason: Prime is the only set whose beam variants are all positive, and a direction seen on one of three sets is a hypothesis, not a result; WebQSP and MAG are the two other sets of the same cell.

**Priors (before any number of the two sets).**

- **D7 (WebQSP).** No variant is ABOVE `control`, and at least one beam variant is BELOW (its giant answer sets hold their golds in the first hops, and a frontier of 100 drops the rows of the other kept nodes).
- **D8 (MAG).** `h3b100` is not BELOW `control` and its paired difference is above 0 (the direction of Prime repeats), and no variant is ABOVE by the +1.0 line.
- **D9 (the three three-hop sets together).** A beam variant is a candidate for a whole-split declaration only if it is ABOVE on at least two of Prime, WebQSP and MAG and BELOW on none; expected NOT met (Prime's `h3b100` has its lower end at +0.27, below the +1.0 line).
- **D10 (size and gate).** Mean pool sizes of every variant are within 5 percent of `control`'s on both sets; `control` equals the engine on every question (0 mismatches).

Each prior is judged against every value of its sentence; results go into this section as section 7.1 after the rows end. Nothing is adopted.
