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

## 6. Results

None yet.
