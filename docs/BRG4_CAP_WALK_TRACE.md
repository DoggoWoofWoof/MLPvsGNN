# BRG4: why MetaQA's capped walk misses its golds (BRG1's trace for a per-node-cap walk)

Declared 28 Nov 2026 19:19 IST, after BRG1 to BRG3 (`docs/BRG1_BRIDGE_TRACE.md`, `docs/BRG2_ORDER_AND_WINDOW_SCREEN.md`, `docs/BRG3_DEPTH_AND_BEAM_SCREEN.md`) and before any number of this page. Read-only diagnostic on a subset; nothing is adopted and no cell or rule of the filed work changes.

## 1. Why

DRP1 section 17 (whole splits, e3v, dense-only, no K, no name seeds) leaves MetaQA at 93.3 within 2,500 nodes, below the 95 target; it is the one in-domain set below it whose cell is not budget-following, so BRG1's trace (which reads budget-following walks only) did not cover it. MetaQA's cell is `rrf1000_b800+STRUCT:h3_c25_v2000`: the first 800 of the fused list, then three hops over the structural families, each frontier node contributing the first 25 neighbours of its row (no cosine inside the cap), the visited cap 2,000 keeping the first fresh nodes of a hop in arrival order. The four sets that meet the target (2Wiki, HotpotQA, SQuAD, MultiHop-RAG) are not traced; the sets below it are, with this page, all traced.

## 2. What runs

`outputs/mp_unified/zbrg4.py` (new; imports `zbrg1.py` and the pool code, edits none). `trace` re-runs `zpl4g.expand4` for the declared cell hop by hop with every hop's ordered candidate list kept and gives every gold of every question one of the reasons of BRG1 (IN_HEAD, IN_EXP, NO_IN_EDGE, BUDGET_CUT = a candidate past the visited cap, WINDOW_CUT = beyond the 25-neighbour cap in a frontier node's row, UPSTREAM_CUT, NO_PATH_IN_H), the BFS distance of each missed gold, the state of its parents (expanded, head only, kept at the last hop, absent), and the dense list's depth of each missed gold; `read` aggregates (`zbrg1.aggregate`, window = 25). The cell is the dense-only arm's, e3v, no K, no SPLADE, no name seeds.

**Subset.** BRG1's: the first 3 chunks of the seeded order of the set (128 questions each). **Gates.** G-C0: the declared cell is the best within 2,500 nodes of the filed whole-split read (`outputs/drp1e/read_metaqa.json`). G-D1 (imported). **G-C1**: the traced walk equals the engine's expansion and grade on every question of every chunk (a mismatch stops the chunk and no number of the set is read). The self-test (the traced walk equals `zpl4g.expand4` on 120 toy walks over six settings, with `vr`, `rr` and `eq`) passed before this page was committed.

## 3. Priors (written before any number of this page)

- **M1.** BUDGET_CUT is the largest reason among the missed golds and holds at least 40 percent of them (a three-hop fan-out over a movie graph overruns a 2,000-node visited cap).
- **M2.** NO_IN_EDGE plus NO_PATH_IN_H hold at most 20 percent of the missed golds (MetaQA's answers are within three hops of the question's entity and the dense seeds find that entity).
- **M3.** WINDOW_CUT (a gold beyond the first 25 neighbours of a frontier node) holds at least 15 percent of the missed golds (director and actor rows are longer than 25).
- **M4.** Questions with four or more golds hold at least 60 percent of the incomplete questions.
- **M5.** The subset's ALL is within 3.0 points of the whole split's 93.3.
- **M6.** G-C1 holds: 0 mismatches on every chunk.

## 4. What the read may say

- Each prior is judged against every value of its sentence.
- Reasons are a diagnosis of the cell, not a lever: a cap or budget change is a new rule and needs its own declaration (and, as every rule, all ten datasets).
- Nothing from SPLADE, K, name seeds or BM25 enters. The hub line stays closed (8.15).

## 5. Rows

CPU only, on the host through one `rx exec` stream (`pylib_run.py outputs/mp_unified/zbrg4.py trace --ds metaqa --host`, then `read` on the laptop).

## 6. Results (28 Nov 2026 19:21 IST)

**What ran.** `trace` on MetaQA, three chunks (384 questions, 2,901 golds; 7.55 golds per question on average, up to 100), module sha256 prefix `9de9a11c451fda46` (local = host), one host stream, about 1 second per chunk. G-C0 and G-D1 PASS; G-C1 PASS on all 3 chunks (0 mismatches). Records: `outputs/brg4/metaqa/gate_trace.json`, `trace_c*.npz`, `outputs/brg4/read_metaqa.json`. Subset read (a screen).

**The pool.** ALL 93.75 percent (360 of 384 questions complete, Wilson [90.9, 95.8]) at 2,509 mean nodes; the whole split's filed read of this cell is 93.3 at 2,485. Golds: 9.4 percent in the 800-deep head, 85.9 percent added by the walk, 4.7 percent missed (136 golds in 24 incomplete questions).

**Table 1. Why the 136 missed golds are missed.**

| reason | golds | share of the missed |
| --- | --- | --- |
| UPSTREAM_CUT (a path of at most three hops exists; the gold's parent never entered a frontier) | 84 | 61.8% |
| WINDOW_CUT (the gold sits beyond the 25th neighbour of a frontier node's row) | 28 | 20.6% |
| BUDGET_CUT (the gold was a hop-3 candidate past the visited cap) | 24 | 17.6% |
| NO_IN_EDGE, NO_PATH_IN_H | 0 | 0.0% |

BFS distance of the missed golds: 1 hop 1, 2 hops 22, 3 hops 113. Hop sizes (means over the 384 questions): hop 1 keeps 32.8 nodes of 32.8 candidates (five seeds, 25 neighbours each at most); hop 2 keeps all 343.7 candidates against a budget of 1,962; hop 3 has 2,171 candidates against a budget of 1,619 and keeps 1,397 (the cap of 2,000 visited nodes binds only here). Among the 24 hop-3 budget cuts the rank over the budget has median 1.10 and 90th percentile 1.37: all are within 2 times the budget but one, all within 4 times. Parents of the missed golds: expanded 38.2 percent, in the head only 14.0, kept at the last hop 2.2, none in the pool 45.6; no missed gold has another gold as a parent. Only 4.4 percent of the missed golds are in the dense list within the first 50 nodes (11.0 percent within 100); the question's seeds hold a gold in 1.3 percent of the questions (5 of 384). By gold count: 1 gold 105 of 105 complete, 2 golds 70 of 73, 3 golds 58 of 58, 4 or more 127 of 148 (85.8 percent at 2,642 mean nodes); 21 of the 24 incomplete questions have four or more golds. Sole blockers of the 24 incomplete questions: upstream 7, budget 6, window 6, the other 5 have two or more reasons.

**Calls on the priors.**

- **M1 NOT HELD.** BUDGET_CUT is 17.6 percent of the missed golds (at least 40 percent expected) and not the largest; UPSTREAM_CUT is 61.8 percent.
- **M2 HELD.** NO_IN_EDGE plus NO_PATH_IN_H hold 0 percent (at most 20 expected).
- **M3 HELD.** WINDOW_CUT holds 20.6 percent (at least 15 expected).
- **M4 HELD.** 21 of the 24 incomplete questions (87.5 percent; at least 60 expected) have four or more golds.
- **M5 HELD.** The subset's ALL, 93.75, is 0.45 above the whole split's 93.3 (within 3.0).
- **M6 HELD.** 0 mismatches on all 3 chunks.

**What the trace says (the reading of the means is mine: UPSTREAM is not traced per gold).** MetaQA's misses are the row cap, not the budget and not the graph: none of the 136 missed golds lacks a path of three hops or fewer over the structural families, hops 1 and 2 never reach the visited cap on average (343.7 kept of a budget of 1,962 at hop 2), so a gold whose parent never entered a frontier lost it to the 25-neighbour cap at hop 1 or hop 2, which together with the 20.6 percent cut by the cap directly makes about 82 percent of the missed golds cap-related, against 17.6 percent for the visited cap at hop 3. The golds are numerous (7.55 per question) and the incomplete questions are the ones with four or more; the question's cosine is not an available key here (the cell has no `cs` and no `vr`: PL's grid found cosine inside the cap hurts on MetaQA, CR1's reversal). Nothing is adopted; what a larger cap would give is the question of Amendment 1.

## 7. Amendment 1 (28 Nov 2026 19:21 IST): what a larger per-node cap gives MetaQA

Declared after section 6 and before any number of the sweep. The trace says about 82 percent of MetaQA's missed golds are cap-related (the first 25 neighbours of a frontier node, in row order), so the question is what the cap costs and what a larger one buys at the same visited cap. A rule needs all ten datasets and a cap sweep on the others is a separate declaration; this amendment is MetaQA's diagnosis.

**What runs.** A new command `capsweep --ds metaqa` of `zbrg4.py` (added after this commit, before any row runs it; self-test first): the declared cell's walk re-run with the per-node cap (hop 1 and later hops together) at 25 (the control), 50, 100 and 200, the visited cap 2,000, the hops 3, the key and the families as declared (row order, no `cs`, no `vr`), on the traced subset (the same three chunks). Per question and variant: the golds in the pool (800-deep head and the walk), the pool size, and the ceiling (golds in the head or among any hop's candidates: a perfect order inside the visited cap). **Gate G-C2:** the cap-25 variant equals the engine's expansion and grade on every question of every chunk. The read gives ALL, golds in the pool, mean nodes, the ceiling, the paired difference to the control with a 2,000-resample bootstrap over questions (seed 20261128) and the `zdrp2.gain_call` at the 1.0-point line. A whole-split read of the best variant follows only if one variant is ABOVE on the subset and not larger than 2,500 nodes in mean (declared here; the same code over all chunks; the best cell's name is then the declared control of a later universal declaration, not an adoption).

**Priors (before any number of the sweep).**

- **C1.** At least one of the caps 50 and 100 is ABOVE the control (82 percent of the missed golds are cap-related; the visited cap binds only at hop 3).
- **C2.** The mean pool size of every variant is within 3 percent of the control's (the visited cap of 2,000 plus the 800-deep head bounds it).
- **C3.** The ceiling is non-decreasing in the cap (25 up to 200).
- **C4.** The cap of 200 is not better than the cap of 100 by more than 0.5 point (more candidates per node meet the visited cap earlier).
- **C5.** The cap-25 variant equals the engine on every question (0 mismatches).

## 8. Results of Amendment 1

None yet.
