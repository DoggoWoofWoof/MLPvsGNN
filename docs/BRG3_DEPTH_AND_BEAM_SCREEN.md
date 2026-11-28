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

## 6. Results

None yet.
