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

## 8. Results of Amendment 1 (28 Nov 2026 19:24 IST)

`capsweep` on MetaQA, the same three chunks (384 questions), module sha256 prefix `c44eaab7a571f767` (local = host; committed 14fd410b), G-C2 PASS on all 3 chunks (0 mismatches). Records: `outputs/brg4/metaqa/gate_cap.json`, `cap_c*.npz`, `outputs/brg4/read_cap_metaqa.json`. Subset read (a screen).

| per-node cap | ALL | golds in the pool | mean nodes (over cap 25) | ceiling ALL (golds among the head and every hop's candidates) | minus cap 25 [95% interval], call |
| --- | --- | --- | --- | --- | --- |
| 25 (the cell) | 93.8 | 95.3 | 2,509 | 95.3 | |
| 50 | 93.0 | 90.5 | 2,623 (1.045) | 100.0 | -0.78 [-3.39, +1.82] LEVEL |
| 100 | 81.8 | 66.8 | 2,685 (1.070) | 99.7 | -11.98 [-15.89, -8.59] BELOW |
| 200 | 74.5 | 53.0 | 2,712 (1.081) | 92.4 | -19.27 [-23.44, -15.10] BELOW |

By gold count (ALL): cap 25 100.0 / 95.9 / 100.0 / 85.8 for 1, 2, 3, 4 or more golds; cap 50 100.0 / 95.9 / 98.3 / 84.5; cap 100 97.1 / 91.8 / 94.8 / 60.8; cap 200 95.2 / 86.3 / 91.4 / 47.3.

**Calls on the priors.**

- **C1 NOT HELD.** Cap 50 is -0.78 [-3.39, +1.82] LEVEL and cap 100 is -11.98 BELOW; neither is ABOVE.
- **C2 NOT HELD.** Mean pool sizes are 1.045, 1.070 and 1.081 times the control's (within 3 percent expected).
- **C3 NOT HELD.** The ceiling is 95.3, 100.0, 99.7 and 92.4: it rises to cap 50 and falls after.
- **C4 HELD.** Cap 200 (74.5) is 7.3 points below cap 100 (81.8), not better.
- **C5 HELD.** 0 mismatches on all 3 chunks.

**What it says.** A larger cap does not buy the golds: at cap 50 every gold is among the candidates of the hops (ceiling 100.0, up from 95.3) and the pool still holds 93.0, and the larger the cap the more the first-arrived nodes of hop 3 are the wrong ones (cap 100: 66.8 percent of the golds in the pool against a ceiling of 99.7). The visited cap keeps the first fresh nodes of a hop in arrival order, and a larger cap fills it with the first rows' neighbours. **MetaQA's misses are an ordering problem inside the visited cap, as Prime's and WebQSP's are**: at cap 50 a perfect order would give 100.0 against 93.0 at 4.5 percent more nodes (the cell's own cap of 25 has a ceiling of 95.3 against its 93.8). The cosine key the other sets use is not an option here (PL's grid found it hurts on MetaQA); Amendment 2 asks whether a question-free key (the frontier's agreement) orders the hop better than arrival. Nothing is adopted.

## 9. Amendment 2 (28 Nov 2026 19:24 IST): which order inside the visited cap

Declared after section 8 and before any number of this amendment. A new command `ordsweep --ds metaqa` of `zbrg4.py` (added after this commit, self-test first): the declared cell's walk at the caps 25, 50 and 100 with the hop's fresh candidates ordered by `control` (arrival, the cell's), `sup` (the number of distinct frontier nodes whose row lists the node, most first, ties to arrival), `ppr` (the sum over those frontier nodes of 1 / (RRF_C + the frontier node's place + 1), ties to arrival) or `vr` (the question's cosine, highest first, ties to the lower id): 12 variants, the control being cap 25 with arrival. Same subset, visited cap 2,000, hops 3, gates and statistics as sections 7 and 8 (**G-C3**: the control equals the engine's expansion and grade on every question). Ceiling as in section 8 per variant. A whole-split read of the best variant follows only if one variant is ABOVE on the subset with a mean pool size no more than 2 percent above the control's.

**Priors (before any number of this amendment).**

- **K1.** At the cap of 25, no order is ABOVE the control (arrival is row order, which already puts a node's neighbours in a fixed relation order, and the other orders cannot add a signal the row does not have).
- **K2.** At the cap of 50, `sup` or `ppr` is ABOVE the control (the golds are all candidates there and the answers of a multi-hop question are reached from several frontier nodes).
- **K3.** `vr` is BELOW the control at the caps 50 and 100 (PL's grid and CR1: cosine hurts on MetaQA).
- **K4.** Every variant's mean pool size is within 10 percent of the control's.
- **K5.** G-C3 holds (0 mismatches).

## 10. Results of Amendment 2 (28 Nov 2026 19:27 IST)

`ordsweep` on MetaQA, the same three chunks (384 questions), module sha256 prefix `bcd149132d7c4611` (local = host; committed 7947be85), G-C3 PASS on all 3 chunks (0 mismatches). Records: `outputs/brg4/metaqa/gate_ord.json`, `ord_c*.npz`, `outputs/brg4/read_ord_metaqa.json`. Subset read (a screen).

| per-node cap | order | ALL | golds in the pool | mean nodes (over the control) | ceiling ALL | minus the control [95% interval], call |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | arrival (the cell) | 93.8 | 95.3 | 2,509 | 95.3 | |
| 25 | `sup` | 92.4 | 92.1 | 2,506 (0.999) | 95.3 | -1.30 [-3.12, +0.26] LEVEL |
| 25 | `ppr` | 93.0 | 93.4 | 2,507 (0.999) | 95.3 | -0.78 [-2.34, +0.78] LEVEL |
| 25 | `vr` | 84.4 | 85.5 | 2,486 (0.991) | 95.3 | -9.38 [-12.24, -6.25] BELOW |
| 50 | arrival | 93.0 | 90.5 | 2,623 (1.045) | 100.0 | -0.78 [-3.39, +1.82] LEVEL |
| 50 | `sup` | 89.1 | 83.6 | 2,614 (1.042) | 100.0 | -4.69 [-7.81, -1.56] BELOW |
| 50 | `ppr` | 89.1 | 86.8 | 2,617 (1.043) | 100.0 | -4.69 [-7.81, -1.82] BELOW |
| 50 | `vr` | 79.9 | 76.2 | 2,567 (1.023) | 100.0 | -13.80 [-17.45, -9.90] BELOW |
| 100 | arrival | 81.8 | 66.8 | 2,685 (1.070) | 99.7 | -11.98 [-15.89, -8.59] BELOW |
| 100 | `sup` | 85.4 | 73.0 | 2,673 (1.065) | 99.7 | -8.33 [-11.72, -5.21] BELOW |
| 100 | `ppr` | 85.2 | 76.0 | 2,679 (1.068) | 99.7 | -8.59 [-11.98, -5.21] BELOW |
| 100 | `vr` | 75.3 | 62.0 | 2,591 (1.032) | 99.7 | -18.49 [-22.40, -14.32] BELOW |

**Calls on the priors.**

- **K1 HELD.** At the cap of 25 `sup` (-1.30) and `ppr` (-0.78) are LEVEL and `vr` (-9.38) is BELOW; none is ABOVE.
- **K2 NOT HELD.** At the cap of 50 `sup` and `ppr` are -4.69 BELOW (an ABOVE was expected).
- **K3 HELD.** `vr` is BELOW at the caps of 50 (-13.80) and 100 (-18.49) (and at 25, -9.38).
- **K4 HELD.** Mean pool sizes are 0.991 to 1.070 times the control's (within 1.10).
- **K5 HELD.** 0 mismatches on all 3 chunks.

**What it says.** No variant is ABOVE the control, so the whole-split read declared for an ABOVE variant does not run, and **MetaQA's cell, the first 25 neighbours of a row in arrival order, is the best of the twelve (cap and order) combinations**: a larger cap, the frontier's agreement (`sup`, `ppr`) and the question's cosine (`vr`) all lose, `vr` by 9 to 18 points. The ceiling at the cap of 50 is 100.0 (every gold a candidate) against 93.0 reached, a headroom of 7.0 points that none of the parameter-free keys reaches. This is the result of BRG2 and BRG3 on a pure-entity set: the candidates contain the golds and the orders do not pick them. Nothing is adopted.

**What the four pages say together (BRG1 to BRG4, subset screens on the six sets below 95).** MuSiQue is missing links (72 percent of the missed golds have no path in two hops). Prime, MAG, WebQSP and MetaQA lose their golds to the order inside a budget: the golds are candidates (ceilings 9.8, 5.0, 5.4 above ALL on the first three, 7.0 at cap 50 on MetaQA), and the order keys tried (arrival or row order, the question's cosine, frontier support, a ppr-style sum, their fusions, a wider window, more hops, a beam of 100) are LEVEL or BELOW the cell's own on every set. Amazon is a first-stage lexical gap (+2.5 at 2,500 nodes for the fused first stage) plus window cuts. What is left for the four order sets is a scorer that sees the question and the edge together, trained on the golds: the stage after the rules.
