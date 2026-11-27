# STK3: do the seven's pool cells carry to the STaRK knowledge bases

**Declared 15:21 IST, 27 Nov 2026, before any number of it.** Owner: the retrieval session. Companion pages: `docs/STK1_STARK_PACKAGE.md` (the package, the encode, the first stage, section 10.1), `docs/STK2_STARK_LINKS.md` (the links), `docs/PL7_COVERAGE_RESTRICTION.md` and `docs/PL1_POOLS_ON_UL2_LAYOUT.md` (the pools and the layout this page reuses), `docs/ING1_INGESTION_AND_STAGE_REGISTRY.md` (the loader).

## 0. The user's words, and the rulings that bind

- 26 Nov: "deal with all the datasets the same way no matter what", "all 10": the three STaRK knowledge bases get the seven's rule, the same code and the same gates, run order by size only.
- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts"; "the pools or the rules or links all should be universal"; the node bound is brought toward 2,500 under ONE rule.
- 26 Nov: pools first, parameter-free; **nothing here trains, no model is fitted, no LLM or spaCy is used, and every pool function is gold-free** (the golds only grade).
- 27 Nov: "keep the queue full"; subset first, then the full read; the user's standing "yeah sure then we can put it to a generalization test".

## 1. The question, and what the first stage already says

STK1 section 10.1 filed the first stage on the test split: the fused list's ALL@2,500 is **64.7 (Prime), 83.9 (MAG), 90.6 (Amazon)**, that is 30.3, 11.1 and 4.4 points short of 95 (target of the repository). On six of the seven the picked cell is a walk over the links from the first stage's seeds (SQuAD's pick is the flat fused top 15). **STK3 asks whether that lift carries to a knowledge base the cells were not chosen on.** The answer is read in two steps:

- **Step A (this page, section 2 on): a transfer read.** The cells that the seven picked for themselves (and the flat first-stage ladder) are run unchanged on the STaRK test questions. **Nothing is selected, tuned or refit on a STaRK question or its dev split.** A cell that does well on Prime did so without having seen Prime.
- **Step B (not declared here): rule R7(2,500)** (PL7 section 4: the smallest dev nodes with dev ALL at least 0.95, else the highest dev ALL, over the one cell table) run on the STaRK dev splits. It needs the table's cost on a STaRK graph, which Step A's time probe prices.

## 2. What is fixed (by reference)

The engine is PL2's to PL6's (`zpl2g.py` to `zpl6g.py`, imported, edited nowhere): the walks, `cs`, `vr`, `sp`, `rk`, `fu`, `rr`, `eq`, the budget-following walk, the cut, the seeds (dense top 5 with SPLADE top 5), the first-stage lists (top 1,000 of each leg, RRF constant 60), the measures (`all_gold_at_pool`, `candidates_mean`), the Wilson interval on ALL, the paired bootstrap of 1,000 resamples. Named cells are built by `Z5.Plan5(names)` and run by `Z5.evaluate5`, as LEG1-A6's module does (`zleg1p.py`).

**The layout** is PL1's SRK layout read in PL1's way (`zpl1m.tables`, `zpl1m.gates`; the MHR precedent `zpl1h.cmd_layout`): the structural store = S with W2_S (label 0 S, label 1 R; S wins a pair both hold), the ner store = F0T with its **order weights**, the knn store = K without the pairs the other two hold; the families are `M3A.FamilyCSR` of those tables, so neighbours are ordered by weight as the seven's are.
STK2 filed F0T as keys only (no weights), so the weights are **rebuilt in process by the seven's own rule** (`zul1.build_entries`, `zul1.forms_from_entries`, `d83stores.form_pairs`, on the same titles and T1 texts as STK2, exactly as `zpl1h.f0t_in_process` does for MultiHop-RAG). No STK2 file is edited.

**The lists and questions.** The package of the KB through `zworld.load`; the first stage's `outputs/stark/<kb>/first.npz` (`top_dense_e3v`, `top_splade`, first 1,000 of each, rows = the test split's query rows); the question and node vectors are e3v's (the encoder arm of the package); golds = the distinct gold positions (STK1 A1). Question order = the test split's row order.

## 3. Gates (a failure stops that KB's reading; none of its numbers is read)

- **G-F (the layout):** the rebuilt F0 keys equal STK2's `links/F0.npz`, the rebuilt F0T keys equal `links/F0T.npz`, every F0T weight is positive, the triple's keys equal the keys; `zpl1m.gates` PASS (knn disjoint from structural and ner, ner disjoint from S, labels, union); the layout's union count equals STK2's store size (`links.json`: 4,431,605 pairs on Prime); the S count equals `links.json` S; the per-family row order equals `M3A.FamilyCSR`'s on the same pairs.
- **G-S (the source):** the engine's cells `dense_top20`, `dense_top100`, `splade_top200` give the same all-gold booleans as the same tops read from `first.npz` and the benchmark's golds, on every question of the first chunk (exact); the cell `rrf1000_b20` and `rrf1000_b100` agree with the fused list's top 20 and top 100 as sets on at least 99 percent of the first chunk's questions (the fused list in `first.npz` is built from lists of depth 2,500, the engine's from depth 1,000; the number is reported).
- **G-P (the probe):** a two-question run of the cell list writes seconds per question, nodes scored and peak memory, **no label is read** (PL6 section 6's probe), before any chunk.

## 4. The cells (fixed at this commit; nothing is added after a number is read)

`zstk3.py cells` reads, from the filed `outputs/pl6/<d>/select.json` of the seven (`d` in squad, musique, hotpotqa, 2wiki, metaqa, webqsp, multihoprag), **every pick named under `picks` and the `incumbent`** (R95 and R98, all, bounded and rung 0), keeps those whose filed mean nodes on their own dataset are **at most 16,000** (the cells above that are WebQSP's and MuSiQue's V = 20,000 to 40,000 walks, which the 2,500 bound already rejected), and adds **the flat ladder of the table**: every `rrf1000_bN`, `dense_topN` and `splade_top200` cell of the table. It writes the list and its hash to `outputs/stk3/cells.json` before any run. The list is the same for all three KBs. **Expected size: 16 to 24 pick cells and 44 flat cells.**

## 5. Population, order, parts

The test split's every question (Prime 2,801; MAG 2,665; Amazon 1,642), in chunks of 128 questions in a seeded permutation (seed 20261127, as `zleg1p.py`), so a prefix of the chunks is a spread sample and a partial read is meaningful. **Subset first:** Prime's first four chunks (512 questions) are read as the screen; the remaining chunks run only if the screen's gates pass and its cost allows. Rows on the host, 2 CPUs each (the engine is single-threaded), parts take every K-th chunk; a part writes each chunk when done (resumable).

## 6. What is read, and the calls

For each cell: ALL (every gold in the pool) with its Wilson 95 percent interval, ANY, mean distinct nodes. **Flat comparison at equal nodes:** for each pick cell the flat cell with the smallest mean nodes at least the pick's (`rrf1000_bN`), the paired difference ALL(pick) - ALL(flat) with the paired bootstrap interval, in points, and the nodes ratio. A pick **LIFTS** a KB if the lower end of that interval is above 0, is **LEVEL** if it contains 0, **BELOW** if the upper end is below 0. The R and K families are judged the way STK2 section 7 fixed it: **by what they add at equal pool size**, pairing a FULL cell with the STRUCT cell of the same walk and budget (where the table has both). No cell is called "the pick for the KB": a read of the best cell is labelled post-hoc, because nothing here selects.

## 7. Expectations, written before any number

- **E1.** G-F and G-S pass on Prime.
- **E2.** The flat fused list's ALL at the seven's bounded sizes rises slowly on Prime (a few points between 500 and 2,500 nodes), as its first-stage ALL@1000 to ALL@2,500 does (59.2 to 64.7).
- **E3.** At least one pick cell within 3,500 nodes LIFTS Prime by at least 5 points over the flat cell of equal nodes (the graph has 62.6 edges per node, so a walk reaches golds the lists do not).
- **E4.** **No cell of this list reaches ALL 0.95 within 5,000 nodes on Prime** (the best leg is 30 points short; the seven needed a pick tuned to each dataset to get to 0.95 and Prime is the hardest first stage).
- **E5.** S carries the lift on Prime: the FULL cell's gain over its STRUCT cell at equal nodes is under 2 points (R-only is 179,072 pairs and K-only 202,891, against 4.05 million of S).
- **E6.** The probe prices the whole list (pick cells and flat cells) at no more than 30 seconds per question on one CPU: a chunk of 128 questions then takes under 64 minutes and Prime's 22 chunks about 23 CPU-hours, about 3 hours on 8 CPUs.

## 8. Not read, and what follows

Not read: any selection on a STaRK dev split (Step B, to be declared with the probe's price in hand); MAG and Amazon until their links rows have finished (`stk2m-links`, `stk2a-links`) and their first stages are filed (done); any new walk, order, cut or window; any model. **Nothing is adopted; no pool is frozen.** The pools read here are on the current links and are provisional in PL7 section 10's sense.

## 9. Code, rows, outputs

- New module `outputs/mp_unified/zstk3.py` (declared here; edit only through an amendment on this page). Commands: `selftest`, `cells`, `layout --kb K`, `probe --kb K`, `run --kb K [--part I N]`, `read --kb K`. It imports `zworld`, `zstk1`, `zstk2` (names only), `zpl1m`, `zul1`, `d83stores` (the F0T weight rule of `zpl1h.f0t_in_process` is copied, not imported: that module reads MultiHop-RAG's files at import) and `zpl2g`..`zpl6g`; it edits none of them. It adds the one thing the engine does not have: a source for a STaRK KB (`load_source` of `zpl2g` is called through a thin wrapper, the engine is not patched).
- Rows (host, classes `stk3p`, `stk3m`, `stk3a`): `stk3p-layout` (CPU, one row, 8 GB asked, the measured peak of STK2's Prime row was 2.3 GB), `stk3p-probe`, `stk3p-run0` to `stk3p-run3` (2 CPUs, 12 GB each; the memory is re-asked from the probe), `stk3p-read`. Outputs `outputs/stk3/<kb>/` (layout.npz and layout.json, gate and probe records, chunk files, read file) and `outputs/stk3/cells.json`.
- **Disclosed:** the cells were chosen on the seven's dev carves and graded on their eval carves; none was chosen on a STaRK question, but the flat ladder and the pick list were read before this page was written (the seven's records), and the e3v first stage of the KBs was read (STK1 section 10.1) before this page. The cell names come from five engine generations (PL2 to PL6); the same engine code runs them all.
