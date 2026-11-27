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

## 10. Amendment 1 (15:27 IST, 27 Nov 2026; before any row of this page has run, no STaRK pool number read)

`zstk3.py` is written (selftest 11 of 11 on the laptop; script sha256 `eebb9f455fec`) and the cell list is filed (`outputs/stk3/cells.json`, cells hash `4b9bfd160b18`). Four things in sections 3 to 6 are made exact here, all before a number exists:

- **The cell list, as built.** 16 pick cells (the five picks and the incumbent of PL6's `select.json` of each of the seven: 19 distinct names, 3 left out above 16,000 filed nodes: `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu`, `rrf1000_b400+FULL:h3_bf_v40000~cs-vr`, `rrf1000_b1600+FULL:h3_bf_v40000~cs-vr`), **9 twins** (the same walk over the other family label, FULL for STRUCT or STRUCT for FULL, where the same table holds it under the cap; section 7's E5 reads them, and they are added to the list now so that E5 can be read at all) and 38 flat cells (`rrf1000_bN`, `dense_topN`, `splade_top200` of the tables; SQuAD's pick `rrf1000_b15` is a flat cell and is counted among the picks): **63 cells**. The "44 flat cells" of section 4 counted every table name without a walk, `frozen_union` and the `equal_rrf_budget_*` cells included; those are not in the list. The expected number of pick cells (16 to 24) held. A filed node count is the dev mean of PL6's table.
- **The flat comparison of section 6 is read against the first stage's own fused list.** The reference of a cell is the top N of the fused list of `first.npz` (depth 2,500), N = the cell's mean nodes rounded up and capped at 2,500; its all-gold booleans are exact at any N and are not limited by the engine's depth-1,000 ladder (which ends near 2,000 nodes). When a pool is larger than 2,500 nodes the reference is shorter than the pool, the record says so (`reference_shorter_than_pool`), and the difference is then an upper bound of the lift. The calls LIFTS, LEVEL and BELOW are read from the paired bootstrap interval of the difference (`zpl2g.paired`, 1,000 resamples, seed 20261125 with the KB's index).
- **G-S** is run on the first chunk of the KB's declared order (128 questions): the cells `dense_top20`, `dense_top100` and `splade_top200` equal the lists of `first.npz` question by question (exact, and their pool sizes equal k); the fused top 20 and top 100 of the engine's depth-1,000 lists equal `first.npz`'s as sets on at least 99 percent of the chunk's questions (the set test is run on the fused rows directly, since the engine's cells return counts, not sets); the all-gold agreement of `rrf1000_b20` and `rrf1000_b100` is written beside it.
- **G-P** runs the whole list on the first two questions with an empty gold list given to the grader, so no label is read; it records seconds per question, every cell's pool size and the peak memory.

**Order and parts.** The chunk order is a seeded permutation (seed 20261127, the KB's index 0, 1, 2 for Prime, MAG, Amazon), as `zleg1p.py`; the screen of section 5 is `run --upto 4` (the first four positions of the order). `ref.npz` (the question ids, the gold counts and the position of the last gold in each first-stage list) is written by the first run row and is what `read` needs, so the read runs on the laptop without the package. A read with fewer chunks than the KB's is labelled PARTIAL.

**Not changed:** the expectations E1 to E6, the cells' rule, the gates' thresholds, the 128-question chunk, and the rule that nothing is selected on a STaRK question.

## 11. Amendment 2 (15:30 IST, 27 Nov 2026; one row failed on the host before it read any data)

Row `stk3p-layout` (15:29, 22 seconds) stopped at `import numba`: the host's environments have no numba, and the seven's pool rows run as `python outputs/host_ops/pylib_run.py <script> ... --host` (numba from the unpacked zip, the package mirror applied). `zstk3.py` had no `--host` flag. **The only change** is that flag on `layout`, `probe` and `run` (it sets `HOSTMODE`, which `plain_op` passes to `Z6.prepare`; the same call as `zleg1p.py`'s). Nothing about the cells, gates, thresholds or expectations changes, and no data had been read (the log's last line is the import error). `stk3p-probe` behind it was dropped by name; both rows are re-added as `stk3p-layout2` and `stk3p-probe2` (the same commands behind `pylib_run.py` and with `--host`); the failed row may be re-sent once by the feeder with the same command and fails the same way. The self-test still passes 12 of 12 (laptop). Script sha256 of the amended module: see the record the layout row writes.

## 12. Amendment 3 (15:37 IST, 27 Nov 2026; G-S failed once on its fused clause; layout and probe read; still no retrieval number)

- **Layout (row `stk3p-layout2`, 13 seconds, peak 0.45 GB; `outputs/stk3/prime/layout.json`): G-F and G-L PASS.** The rebuilt F0 and F0T keys equal STK2's files (183,546 and 183,735 pairs), every F0T weight is positive, S is 4,049,642 pairs as `links.json` has it, the five links files equal their record's sha256, the families are disjoint where the dedup says, and **the union is 4,431,605 pairs, STK2's store**; the row order of the ner and knn stores equals `FamilyCSR`'s. The walk stores: structural 4,076,842 pairs (S 4,049,642, R 27,200), ner 159,357 (24,378 of F0T's pairs are S pairs and leave), knn 202,891 (76,360 of K's pairs lie in S or R and leave; 202,891 is STK2's K-only count).
- **Probe (row `stk3p-probe2`; `outputs/stk3/prime/probe.json`; no label read): 5.6 and 2.95 seconds for the two questions over the 63 cells, peak 0.57 GB, the largest pool 15,400 nodes** (the MuSiQue pick's cut). E6 (at most 30 seconds a question) is met by a factor of five or more; the run rows ask 4 GB, not the 12 GB the probe row had.
- **G-S failed, and the failure is the gate's design, not the data.** On the gate chunk (chunk 10 of Prime's order, 128 questions) the three exact clauses hold (`dense_top20`, `dense_top100`, `splade_top200`: 0 mismatches, sizes equal k) and the all-gold booleans of `rrf1000_b20` and `rrf1000_b100` equal those of the fused top 20 and top 100 of `first.npz` on **128 of 128 questions** (agreement 1.000 and 1.000), but the **set** test of section 3 read **0.820 (top 20) and 0.305 (top 100)** against the declared 0.99, so the rows `stk3p-run0` and `stk3p-run1` stopped before any chunk (record kept as `outputs/stk3/prime/gate_s.FAIL1.json`). The set test assumed what section 3 itself said was not true: STK1's fused list (float32 1/(60 + rank), lists 2,500 deep, ties by node id) and the engine's (M3A's inherited equal-RRF over lists 1,000 deep) are two implementations, and an item in the other leg's ranks 1,000 to 2,500 moves several ranks near the top 100. **Amended before any chunk runs:** the fused clause is the all-gold agreement of the two lists' top 20 and top 100 on the gate chunk, **at least 0.98** (at most 2 of 128 questions); the set agreement is still written, as information. The two fusions are different baselines and the read reports both at equal pool size (`flat_ref` of every cell, including the engine's own flat cells), so a difference between them is read, not assumed away. E1 is read against the amended gate and the first failure stays on this page. The threshold was set after the numbers 1.000 and 1.000 were seen; they are gate quantities (plumbing), not outcome quantities.
- The failed rows are re-sent once by the feeder with the code then on the host (the amended module); no row is added.

## 13. Prime's screen read (16:04 IST, 27 Nov 2026; four chunks, 497 of 2,801 test questions; PARTIAL)

**What was read, and when.** The rows `stk3p-run0` and `stk3p-run1` (after amendment 3; G-S PASS on chunk 10) wrote chunks 10, 16, 18 and 21 of Prime's seeded order (the first four positions; chunk 21 is the last chunk of the split, 113 questions), 497 questions, 3.4 seconds a question on one CPU, peak 0.64 GB. `zstk3.py read --kb prime` was run on them under section 6's rules and its numbers are filed here at once, before any further row ran; the screen's record is `outputs/stk3/prime/read.screen.json`. The screen is an easier sample than the whole split: the fused list's ALL@2,500 is **69.2 on these 497 questions against 64.7 on all 2,801** (STK1 section 10.1), about two standard errors, so the levels below will fall on the whole split and the paired differences are the read. The remaining 18 chunks are queued (`stk3p-run2` to `stk3p-run5`).

**Flat ladder (the engine's own flat cells equal the first stage's fused list).** At 100, 400, 1,000 and 1,806 nodes the engine's `rrf1000_bN` cell reads 46.5, 56.7, 63.2 and 66.8, level with first.npz's fused top N (paired differences +0.2, -0.2, 0.0 and -0.4 points, every interval holding 0). The dense leg alone is **BELOW** the fused list (top 200: -3.4 [-5.8, -1.2]; top 1,000: -4.2 [-6.6, -2.0]) and SPLADE alone further below (top 200: -7.2 [-10.3, -4.6]). The fused list's ALL on the covered questions: @100 46.3, @500 57.9, @1,000 63.2, @2,000 67.8, @2,500 69.2.

**Pick cells against the flat cell of equal nodes** (ALL in percent, Wilson 95 percent interval where given; difference is paired, bootstrap interval; mean distinct nodes; all calls from section 6): **14 of the 16 pick cells LIFT and 2 are LEVEL; none is BELOW.** The two LEVEL cells are the smallest ones (`rrf1000_b15`, 15 nodes, +0.2; `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50`, 75 nodes, +0.4 [-1.8, +2.8]).

| cell | nodes | ALL | flat at equal nodes | difference | call |
| --- | ---: | ---: | ---: | ---: | --- |
| `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 117 | 62.0 | 48.9 | +13.1 [+8.9, +16.7] | LIFTS |
| `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 266 | 70.8 | 54.9 | +15.9 [+12.3, +19.9] | LIFTS |
| `rrf1000_b400+FULL:h1_c100~s20+top800` | 952 | 80.5 | 62.8 | +17.7 [+13.7, +21.3] | LIFTS |
| `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 2,065 | 86.5 | 67.8 | +18.7 [+14.9, +22.5] | LIFTS |
| `rrf1000_b400+STRUCT:h3_c25_v2000` | 2,309 | 75.9 | 68.8 | +7.0 [+3.4, +11.1] | LIFTS |
| `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 2,479 | 88.5 [85.4, 91.0] | 69.2 | +19.3 [+15.7, +22.9] | LIFTS |
| `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 5,778 | 93.8 [91.3, 95.6] | 69.2 (reference is the 2,500 list, shorter) | +24.5 | LIFTS |
| `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | 15,391 | 92.4 [89.7, 94.4] | 69.2 (shorter) | +23.1 | LIFTS |

Above 2,500 nodes the first stage's own list ends, so the flat reference is shorter than the pool and those lifts overstate the equal-nodes gain; at or under 2,500 nodes the comparison is equal.

**FULL against STRUCT at equal walk and budget (the nine twins).** FULL minus STRUCT, in points: +0.4 [-1.2, +2.0] (15,391 nodes), -0.6 (117), +1.0 [-0.2, +2.2] (266), +1.4 [-0.8, +3.4] (5,005), **+2.6 [+0.8, +4.2] (2,300 nodes, `rrf1000_b400+h3_c25_v2000`)**, +0.4 (6,227), -0.4 (5,667), -1.2 [-2.6, 0.0] (2,039) and **-4.0 [-6.4, -1.8] (50 nodes, `rrf1000_b25+h1_c100+top25`: FULL is worse)**; nodes ratios 0.98 to 1.02. The mean of the nine differences is -0.04 points.

**Calls on the expectations (screen only; every call is repeated on the whole split).**

- **E1 PASS** (G-F on `stk3p-layout2`; G-S on chunk 10 after amendment 3, with section 12's disclosure).
- **E2 NOT HELD AS WORDED.** The fused list rises from 57.9 at 500 nodes to 69.2 at 2,500, **+11.3 points, not "a few"**. Its rise from 1,000 to 2,500 nodes is +6.0 on the screen against +5.5 on the whole split (59.2 to 64.7), so the slope the expectation pointed at is as written; the 500-node start was misjudged.
- **E3 HELD.** Fourteen pick cells LIFT; within 3,500 nodes the largest lifts are +19.3 [+15.7, +22.9] at 2,479 nodes and +18.7 at 2,065, against the 5 points expected.
- **E4 HELD on the screen.** The best ALL within 5,000 nodes is **88.5 [85.4, 91.0] at 2,479 nodes**; no cell reaches 0.95 at any size (best of all 63: 93.8 [91.3, 95.6] at 5,778 nodes, 92.4 at 15,391). The question stays open to the whole split, where the screen's easier sample will pull every level down.
- **E5 HELD IN EIGHT OF NINE.** S carries the lift: eight twin gaps are under 2 points, one is +2.6 [+0.8, +4.2] (an exception the expectation does not allow; R and K add there) and one is -4.0 (FULL worse at 50 nodes; the cause is not read).
- **E6 HELD** by a wide margin: 3.4 seconds a question, the whole list; Prime's whole split is about 2.6 CPU-hours (the expectation said 23) and the largest pool is 15,400 nodes.

**What the screen says about the user's target (95 percent of the golds at about 2,500 nodes, one rule on all ten).** The seven's pick cells carry to Prime: the best cell at 2,479 nodes is **19.3 points above the flat list at equal nodes**, and this is a transfer, nothing selected on a STaRK question. It is **6.5 points short of 95** on this easier sample, and 93.8 at 5.8 thousand nodes is the highest reading, so no pick as filed reaches the target. ANY is 93.4 at 2,479 nodes against ALL 88.5: about five percent of the questions have some gold in the pool and not all. The remaining 18 chunks give the whole-split read; **nothing is adopted and no pool is frozen**. Candidates for what follows, none declared: a gold-link diagnosis on Prime's dev split in the way of the seven's (which golds are missed, how far they sit from a seed, whether a link of any family reaches them), and Step B (R7(2,500) on the dev split). Both wait for the whole-split read.

## 14. Prime's whole-split read (16:51 IST, 27 Nov 2026; 22 of 22 chunks, 2,801 questions; COMPLETE)

**What ran.** The rows `stk3p-run2` to `stk3p-run5` wrote the other 18 chunks (queued 16:03; 36 to 44 minutes a part, 2 CPUs each; peak 0.64 GB), so Prime's 22 chunks hold every test question once under one layout sha and one cells sha. **Cost: 3.12 CPU-hours, 4.01 seconds a question** for the whole list. `zstk3.py read --kb prime` gave the numbers below; the record is `outputs/stk3/prime/read.json` and the 22 chunk files. The screen of section 13 was an easier sample, as it said: the paired lifts hold (best cell +19.3 on the screen, +20.9 here) and every level is lower.

**First stage on all 2,801 questions (STK1 section 10.1's reference):** the fused list's ALL @100 43.9, @500 54.6, @1,000 59.2, @2,000 63.3, @2,500 64.7; the dense leg alone 39.1, 51.4, 56.0, 60.4, 61.9. The engine's own flat cells equal it at 100 and 400 nodes (-0.0 and -0.2) and fall **BELOW** it by 0.5 [-0.8, -0.2] at 1,000 nodes and 0.8 [-1.2, -0.5] at 1,811 nodes (the depth effect section 12 gave: the engine fuses lists 1,000 deep, STK1 lists 2,500 deep). The pick cells below are compared with STK1's list, the stronger of the two. Dense alone is BELOW the fused list by 4.0 [-4.9, -3.0] (top 200) and 3.1 [-4.1, -2.2] (top 1,000), SPLADE alone by 7.0 [-8.0, -6.0] (top 200).

**Pick cells against the flat list at equal nodes: 13 of 16 LIFT, 3 LEVEL, none BELOW** (ALL in percent with the Wilson 95 percent interval, ANY, mean distinct nodes, paired difference with its bootstrap interval; `*` = the flat reference ends at 2,500 nodes, shorter than the pool, so the difference overstates the equal-nodes gain). The three LEVEL cells are the smallest: `rrf1000_b15` (15 nodes, -0.0), `rrf1000_b50+KNN:h1_c100+top25` (63 nodes, -0.2 [-0.7, +0.4]) and `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (75 nodes, -0.6 [-1.5, +0.2]).

| cell | nodes | ALL | ANY | flat at equal nodes | difference | call |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rrf1000_b25+FULL:h1_c100+top25` | 50 | 44.8 [43.0, 46.7] | 59.8 | 39.3 | +5.5 [+4.4, +6.7] | LIFTS |
| `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 118 | 59.8 [58.0, 61.6] | 74.5 | 45.6 | +14.2 [+12.6, +15.8] | LIFTS |
| `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 269 | 69.3 [67.5, 70.9] | 82.1 | 51.1 | +18.2 [+16.6, +19.8] | LIFTS |
| `rrf1000_b400+FULL:h1_c100~s20+top800` | 957 | 78.5 [76.9, 80.0] | 89.4 | 58.7 | +19.8 [+18.0, +21.5] | LIFTS |
| `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 2,065 | 84.8 [83.4, 86.1] | 91.0 | 63.3 | +21.5 [+19.9, +23.2] | LIFTS |
| `rrf1000_b400+STRUCT:h3_c25_v2000` | 2,315 | 73.3 [71.6, 74.9] | 84.7 | 64.1 | +9.2 [+7.7, +10.8] | LIFTS |
| `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 2,475 | **85.5 [84.1, 86.7]** | 92.0 | 64.6 | **+20.9 [+19.3, +22.5]** | LIFTS |
| `rrf1000_b25+STRUCT:h3_c100_v5000` | 5,000 | 83.3 [81.9, 84.7] | 90.9 | 64.7 | +18.6 * | LIFTS |
| `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 5,753 | 90.2 [89.0, 91.2] | 94.9 | 64.7 | +25.5 * | LIFTS |
| `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 5,780 | **91.9 [90.8, 92.8]** | 95.0 | 64.7 | +27.2 [+25.6, +28.8] * | LIFTS |
| `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | 15,383 | 90.5 [89.4, 91.5] | 94.9 | 64.7 | +25.8 * | LIFTS |

(The other five pick cells are in `read.json`: the three LEVEL cells named above and two that LIFT, at 5,870 and 6,254 nodes.)

**FULL against STRUCT at equal walk and budget (the nine twins).** FULL minus STRUCT in points: -0.1 [-0.6, +0.4] (15,383 nodes), **-1.0 [-1.8, -0.3]** (121), -0.2 [-0.9, +0.3] (276), +0.8 [0.0, +1.6] (5,004), **+1.1 [+0.4, +1.8] (2,300)**, -0.2 [-0.7, +0.4] (6,231), **-0.6 [-1.1, -0.1]** (5,675), **-0.9 [-1.5, -0.4]** (2,039) and **-4.2 [-5.2, -3.3]** (50 nodes); nodes ratios 0.98 to 1.03; the mean of the nine is -0.59 points.

**Calls on the expectations (whole split).**

- **E1 PASS.**
- **E2 NOT HELD AS WORDED.** The fused list rises from 54.6 at 500 nodes to 64.7 at 2,500, **+10.1 points, not "a few"**; from 1,000 to 2,500 it rises +5.5 (59.2 to 64.7), the figure the expectation was anchored on.
- **E3 HELD.** 13 pick cells LIFT; within 3,500 nodes the largest lifts are +21.5 [+19.9, +23.2] at 2,065 nodes and +20.9 [+19.3, +22.5] at 2,475, against the 5 points expected.
- **E4 HELD.** The best ALL within 5,000 nodes is **85.5 [84.1, 86.7] at 2,475 nodes** (ANY 92.0); no cell of the 63 reaches 0.95 at any size, the highest being 91.9 [90.8, 92.8] at 5,780 nodes and 90.5 at 15,383 nodes. The expectation's "at least 30 points short" for the first stage's best leg was 30.3 points (64.7); the best cell here is 9.5 points short at 2,475 nodes.
- **E5 HELD, nine of nine: every FULL-minus-STRUCT gain is under 2 points** (the largest +1.1 [+0.4, +1.8]); the point estimate is below zero in seven pairs and the interval lies entirely below zero in four (-1.0, -0.6, -0.9, -4.2). **On Prime S carries the lift and the R and K edges add nothing at equal size** (they cost up to 4.2 points at 50 nodes); the screen's +2.6 pair came to +1.1 on the whole split.
- **E6 HELD** (4.01 seconds a question against the 30 allowed; 3.12 CPU-hours against the 23 expected).

**What the whole split says about the user's target (95 percent of the golds at about 2,500 nodes, one rule on all ten).** The seven's pick cells carry to Prime and **lift it by 20.9 points at equal nodes** (flat 64.6, best pick 85.5); that is a transfer, nothing was selected on a STaRK question, and the best cell is read post-hoc (section 6: no cell is "the pick" for Prime). The target is **9.5 points away at 2,475 nodes and 3.1 points away at 5,780 nodes. ANY stops at 95.0 and 94.9 at 5,780 and 15,383 nodes: about five percent of Prime's test questions have no gold in any pool of this list (the largest cell is 15,400 nodes), so ALL cannot reach 0.95 with these seeds (five from each leg) and these walks (three hops over the walk stores); the larger cells read here add nothing to ANY beyond about 5,800 nodes.** What the missing five percent are (golds with no link to the seeds, golds beyond three hops, or golds the seeds' lists never reach) is not read; the test split is the transfer read and nothing is diagnosed on it. **Nothing is adopted and no pool is frozen.** Not declared, in this order: the same transfer read on Amazon and MAG when their links rows finish; a diagnosis of the missing five percent on Prime's dev split, which needs the first stage on that split (an STK1 amendment); Step B (R7(2,500) on the dev split).

## 15. Amendment 4 (16:53 IST, 27 Nov 2026; one descriptive read of the questions Prime's best cells miss, declared before it runs; code `zstk3.py missed`)

**Why.** Section 14 found that ANY stops at about 95 percent in every cell, so about five percent of Prime's test questions have no gold in any pool, and that the best cell at 2,475 nodes holds all golds for 85.5 percent and some gold for 92.0 percent. Whether those losses are single-gold questions the first stage never reaches, or multi-gold questions of which the walks take some golds, decides which lever to declare next. The records to answer it are already on disk (the chunks hold the golds present per question, `ref.npz` the gold counts and the first-stage position of the last gold), so the read needs no row.

**What is read (Prime's 22 chunks, all 2,801 questions).** Three cells, named by section 14's whole-split read and therefore post-hoc: **A** `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (2,475 nodes), **B** `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` (5,780) and **C** `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` (15,383). For each: (1) by gold count (1, 2, 3, 4 or more) the number of questions, ALL, ANY and the mean share of the question's golds in the pool; (2) the questions with ANY = 0 (no gold in the pool): their number, the split by gold count, and the split by the first-stage position of their last gold in the fused list (inside the top 1,000; inside 1,001 to 2,500; some gold outside the 2,500-deep list, which `ref.npz` stores as -1); (3) the questions with some but not all golds: their number and the mean share of golds they miss. The record is `outputs/stk3/prime/missed.json`.

**Expectations (written before any of these numbers; the ANY and ALL totals above are known, the splits are not).**

- **X1.** ALL falls with the gold count in cell A: ALL(1 gold) > ALL(2) > ALL(3) > ALL(4 or more).
- **X2.** In cell C, at least half of the questions with ANY = 0 have a gold outside the first stage's 2,500-deep fused list (the first stage's reach, not the walk's, bounds them).
- **X3.** In cell A, at least half of the questions with ANY = 0 are single-gold questions.

**What this is not.** A descriptive read of the test split, which is the transfer read of section 1: **no cell, rule, walk, cut or window is chosen or changed from it**, nothing is adopted, and any rule it suggests is declared first and read on Prime's dev split or on MAG and Amazon, which this read does not touch. The module change is the command `missed` (laptop only, reads the chunks and `ref.npz`, writes `missed.json`, no per-KB branch) and a selftest check; the rest of `zstk3.py` is not edited.

## 16. The missed questions on Prime, read (16:54 IST, 27 Nov 2026; section 15's read, run after amendment 4 was pushed; descriptive, nothing chosen)

`zstk3.py missed --kb prime` on the 22 chunks (all 2,801 questions; record `outputs/stk3/prime/missed.json`). Prime's test questions are **63 percent single-gold** (1,769 of 2,801; 319 with two golds, 187 with three, 526 with four or more).

| cell | nodes | ALL, 1 gold | ALL, 2 | ALL, 3 | ALL, 4+ | ANY = 0 (questions, share) | of those: single-gold | of those: a gold outside the 2,500 list | partial (questions, share of golds missed) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 2,475 | 88.9 | 85.6 | 88.8 | 72.6 | 225 (8.0 %) | 196 (87.1 %) | 202 (89.8 %) | 182 (41.6 %) |
| B `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 5,780 | 92.5 | 92.8 | 96.3 | 87.6 | 140 (5.0 %) | 132 (94.3 %) | 139 (99.3 %) | 87 (38.2 %) |
| C `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | 15,383 | 92.5 | 92.5 | 90.9 | 82.5 | 143 (5.1 %) | 133 (93.0 %) | 139 (97.2 %) | 123 (37.5 %) |

(For a single-gold question ALL and ANY are the same event, so the first column is also ANY. ANY for two, three and four or more golds in cell A: 96.2, 96.8, 97.9; in cell B: 98.4, 100.0, 99.4.)

**Calls on the expectations.** **X1 NOT HELD:** in cell A ALL reads 88.9, 85.6, 88.8, 72.6 by gold count, not falling at each step; it falls clearly only for four or more golds (the bins hold 1,769, 319, 187 and 526 questions). **X2 HELD:** 97.2 percent of cell C's questions with no gold (139 of 143) have a gold outside the first stage's 2,500-deep fused list. **X3 HELD:** 87.1 percent of cell A's questions with no gold (196 of 225) are single-gold.

**What it says (descriptive; no rule follows from it by itself).**

- **The no-gold questions are single-gold questions whose gold is outside the first stage's list.** At 2,475 nodes 196 of the 225 are single-gold and 202 have a gold outside the list; at 5,780 nodes 132 of 140 and 139 of 140. These are questions the first stage does not reach at 2,500 and the walks do not reach from the seeds in three hops within the budget: a first-stage, seed or link-reach loss, not a pool-size loss (the 15,383-node cell leaves 143 of them, the 5,780-node cell 140).
- **The partial questions carry the rest, and they are many-gold questions.** At 2,475 nodes 182 questions (6.5 percent) hold some golds and miss on average 41.6 percent of them; four-or-more-gold questions reach ANY 97.9 percent and ALL 72.6 percent. Cells B and C show the same at larger size (87 and 123 questions).
- **Cell B (STRUCT, 5,780 nodes) beats cell C (FULL, 15,383 nodes) on ALL for three and four-or-more golds** (96.3 against 90.9, 87.6 against 82.5): the two cells differ in family (FULL against STRUCT), visit budget (20,000 against 5,000) and seed budget (400 against 1,600), so the cause is not isolated here; it is consistent with section 14's finding that R and K add nothing on Prime.
- The target (95 percent at about 2,500 nodes) is therefore missed by 11.1 points on single-gold questions (cell A: 88.9) and by 22 points on questions with four or more golds (72.6), and the test split's single-gold share makes the first the larger part of the loss (196 of 225 no-gold questions; of the 14.5 percent of questions not complete, 8.0 points are no-gold and 6.5 points partial).

**Not read.** How far (in hops, in which family) the missed golds sit from the seeds; whether a path of any length reaches them; whether they are reachable from the first stage's list deeper than five seeds per leg. Those are the next read (section 17, to be declared before it runs); nothing here is adopted.

## 17. Amendment 5 (16:58 IST, 27 Nov 2026; one descriptive read of the golds the best cells miss, by link degree and node type, declared before it runs; code `zstk3.py gold_profile`)

**Why.** Section 16 found the no-gold questions to be single-gold questions whose gold the first stage does not list and the walks do not reach, and that no pool size recovers them. Two things can be read from records on disk without the engine: whether those golds have links at all (a gold with no link of any family cannot be reached by any walk), and what kind of node they are. The first separates an isolation ceiling from a ranking loss; the second says where the first stage and the walks lose.

**What is read (Prime, the 22 chunks, all 2,801 questions; the same three cells A, B and C as section 15).** The single-gold questions (1,769) in three groups per cell: all, **hit** (the gold is in the pool) and **missed** (it is not). For the golds of each group: the number; the share with link degree 0 and the mean and median degree, where a node's degree is the number of distinct pairs of a store it is an end of, counted in **S alone**, in the **structural store** (S and R) and in the **union** (structural, ner and knn stores, which the layout keeps disjoint); and the **node type** table: per type the golds of all single-gold questions, the golds missed, and the miss rate, for the eight largest types. Degrees come from `layout.npz`; types from `nodes.jsonl`; the golds from the test split's query file; the hits from the chunks. The record is `outputs/stk3/prime/gold_profile.json`.

**Expectations (written before any of these numbers).**

- **G1.** Fewer than 2 percent of cell A's missed golds have union degree 0 (they are linked; the loss is ranking and reach, not isolation).
- **G2.** The miss rate of cell A differs by at least 10 points between the two largest gold types.
- **G3.** The median union degree of cell A's missed golds is lower than that of its hit golds.

**What this is not.** The same as section 15: a descriptive read of the test split, **nothing is chosen or changed from it**, nothing is adopted, and any rule it suggests is declared first and read on Prime's dev split or on MAG and Amazon. The change to the module is the command `gold_profile` (host row, numpy only, reads the layout, the chunks, `ref.npz` and the package; writes `gold_profile.json`, no per-KB branch) and a selftest check; the rest of `zstk3.py` is not edited.

## 18. The missed golds on Prime, by link degree and node type, read (16:59 IST, 27 Nov 2026; section 17's read, run on the host after amendment 5 was pushed; descriptive, nothing chosen)

`zstk3.py gold_profile --kb prime` (host sha a10d8e0fb8f59432 = the laptop's; record `outputs/stk3/prime/gold_profile.json`): the 1,769 single-gold questions, the same cells A, B and C. Degrees count the distinct pairs a node is an end of.

| group (single-gold golds) | golds | degree 0 in S | S degree, mean / median | union degree, median |
| --- | ---: | ---: | ---: | ---: |
| all | 1,769 | **0** | 335.2 / 62 | 73 |
| A hit | 1,573 | 0 | 328.8 / 52 | 68 |
| **A missed** (2,475 nodes) | 196 | **0** | 386.4 / 105 | **115** |
| B hit | 1,637 | 0 | 350.6 / 62 | 71 |
| **B missed** (5,780 nodes) | 132 | **0** | 143.9 / 80.5 | **88** |
| C hit | 1,636 | 0 | 291.8 / 54.5 | 68 |
| **C missed** (15,383 nodes) | 133 | **0** | 869.3 / 115 | **121** |

Node types of the single-gold golds, with the golds missed by cell A (rate in percent; cells B and C in `gold_profile.json`): disease 635 golds, 32 missed (5.0); gene/protein 590, 102 (17.3); drug 304, 41 (13.5); pathway 182, 4 (2.2); anatomy 30, 6 (20.0); effect/phenotype 16, 2 (12.5); exposure 6, 6 (100); cellular component 4, 2 (50). Cell B: gene/protein 12.0 (71 missed), drug 8.9 (27), disease 3.6 (23); cell C: gene/protein 12.4 (73), drug 8.9 (27), disease 2.5 (16), anatomy 43.3 (13 of 30).

**Calls on the expectations.** **G1 HELD:** none of the 196 golds cell A misses has degree 0 in the union, and none of the 1,769 single-gold golds has degree 0 even in S alone. **G2 HELD:** disease is missed at 5.0 percent and gene/protein at 17.3, a gap of 12.3 points. **G3 NOT HELD, and the other way round:** the median union degree of cell A's missed golds is **115 against 68 for its hit golds** (88 against 71 in cell B, 121 against 68 in cell C); the means follow in cells A and C (396.6 against 340.1; 878.8 against 303.1) and not in cell B (155.4 against 361.8), so the median is the stable reading.

**What it says (descriptive; no rule follows from it by itself).**

- **There is no isolation ceiling on Prime.** Every single-gold gold is linked in S; the golds the cells miss are linked, and on the median better linked than the golds they find. The loss is the first stage's rank plus the walks' reach and ordering, not a gold that no walk can touch.
- **The misses sit in gene/protein and drug golds.** Of cell A's 196 missed golds 102 are gene/protein and 41 drug (73 percent together), against 33 and 17 percent of all single-gold golds; disease (36 percent of golds) holds 16 percent of the misses and pathway almost none. The rare types (anatomy, exposure, cellular component; 40 golds in all) miss at 20 to 100 percent in cell A. The types are Prime's own, so a type-based rule would not be a universal one; the read says where the first stage and the walks lose on this KB, not which lever to pull.
- **Cell A's and cell C's missed golds have higher degree than their hit golds; cell B's mean is lower.** The median is above in all three; the pools differ by walk, family and budget, so the three groups are not one population.

**Not read.** Why these golds are not ranked: whether the entities a question names are in the seeds, how many of the gold's neighbours the walk visits before the budget ends, and where the gold sits in the walk's own candidate order. Those need the engine's internal candidate lists, which `evaluate5` does not return; reading them needs a declared module change (or a dev-split first stage, an STK1 amendment) and is not declared here. **Nothing is adopted; no pool is frozen.** MAG and Amazon follow when their links rows finish, with the same cells, gates and reads.

## 19. Amendment 6 (18:53 IST, 27 Nov 2026; Amazon and MAG declared before any of their rows runs; no STaRK-Amazon or STaRK-MAG pool number read)

**Why.** Sections 13 to 18 read Prime. Section 8 left MAG and Amazon until their links rows finish, with the same cells, gates and reads. The links rows `stk2a-links` (Amazon) and `stk2m-links` (MAG) are still running (3 h 46 min at 18:50; W2_S done at 16:24 and 15:54), and the host queue behind them is empty: the feeder holds 31 unsent rows, all waiting on held parents, so the host is at about 29 percent CPU with the card idle. The rows below are queued **now with the links rows as parents**, so each KB's layout starts by itself the moment its links row ends, and the expectations are written here before any of these rows can run.

**What runs (no code change; `zstk3.py` a10d8e0fb8f59432 is not edited, no per-KB branch).** For each KB, the same chain as Prime, on the same 63 cells (`outputs/stk3/cells.json`, hash 4b9bfd16...), the same gates G-F, G-L, G-S (as amended in section 12) and G-P, the same chunking (128 questions, seeded order, seed 20261127, KB index 1 MAG and 2 Amazon):

- Amazon (1,642 test questions, 13 chunks): `stk3a-layout` (parent `stk2a-links`; 2 CPUs, 10 GB asked), `stk3a-probe` (parent the layout; two questions), `stk3a-run0` to `stk3a-run3` (parent the probe; `run --kb amazon --part i 4`; 2 CPUs, 6 GB each).
- MAG (2,665 test questions, 21 chunks): `stk3m-layout` (parent `stk2m-links`; 2 CPUs, 16 GB asked), `stk3m-probe`, `stk3m-run0` to `stk3m-run5` (`run --kb mag --part i 6`; 2 CPUs, 8 GB each).

Every part checks the gate on the first chunk of the order before it writes a chunk, so a failed G-S stops the KB with no number read (section 3). The memory is asked above Prime's measured peaks (0.45 GB layout, 0.57 GB probe) because the KBs hold 8 and 14.5 times Prime's nodes; the peaks are read from the layout and probe records and the asks are corrected for any later row. The subset-first step of section 5 is the order itself: a part writes its chunks in the declared order, so any prefix of the chunks is a spread sample; the probe's price is read before the first chunk is, and the run rows are cancelled if the probe prices the list above 30 seconds a question (13.7 CPU-hours on Amazon, 22 on MAG).

**Reads after the chunks.** `read --kb K` (section 6, the same calls), then the two descriptive reads of sections 15 and 17 on each KB by the same commands (`missed`, `gold_profile`) and **the same three named cells A, B and C** (not re-chosen on MAG or Amazon). Those two reads carry no expectation here: Prime's results (X1 NOT HELD, X2 HELD, X3 HELD; G1 HELD, G2 HELD, G3 NOT HELD) are the reference.

**Expectations (written before any Amazon or MAG chunk exists; each can fail).**

- **A1.** G-F, G-L, G-S and G-P pass on both KBs with the same code.
- **A2.** On each KB at least one pick cell within 3,500 nodes LIFTS the flat cell of equal nodes (lower end of the paired interval above 0), and the best such lift is smaller than Prime's +20.9 because the room is smaller: **at most 6 points on Amazon** (STK1's fused ALL@2,500 is 90.6; Prime closed about 59 percent of its room, 59 percent of 9.4 is 5.5) and **at most 10 points on MAG** (83.9; 59 percent of 16.1 is 9.5).
- **A3.** The point estimate of at least one **Amazon** cell within 5,000 nodes is ALL 95.0 or more (borderline: it needs +4.4 over the flat list); **no MAG cell** within 5,000 nodes reaches 95.0.
- **A4.** On MAG the FULL cell's gain over its STRUCT twin at equal nodes is under 2 points in every twin pair (as Prime). **No expectation on Amazon:** its rule family is far larger than Prime's (F0 5.87 million pairs against Prime's 0.18 million, and 647,816 L4 same-name sibling links), so the pairs are read as information.
- **A5.** The probe prices the 63 cells at no more than 30 seconds a question on one CPU on each KB, peak memory below 16 GB.
- **A6.** The engine's flat cells are within 1 point of STK1's fused list at equal nodes on both KBs (Prime: 0.5 and 0.8 below).

**What this is not.** The same as sections 8 and 15: nothing is chosen, adopted or frozen; no cell is the pick of a KB; a read of the best cell is post-hoc. No STaRK dev split is read (Step B stays undeclared). The pools are on the current links and provisional in PL7 section 10's sense.

## 20. Amazon's whole-split read (20:02 IST, 27 Nov 2026; 13 of 13 chunks, 1,642 test questions; COMPLETE; section 19's reads, run as declared)

**What ran.** `stk3a-layout` (330 s, peak 2.34 GB) and `stk3a-probe` (23 s) ran first, then `stk3a-run0` to `stk3a-run3` (12 to 15 minutes a part, 2 CPUs each, queued behind the links row; about 1.7 CPU-hours, 3.8 seconds a question for the whole list, taken from the rows' wall times). One layout sha and one cells sha (`4b9bfd16...`) cover the 13 chunks. `zstk3.py read`, `missed` and `gold_profile --kb amazon` ran on the host after the rows; `outputs/stk3/amazon/`. Gates: layout PASS (G-F, G-L: 12,841,153 pairs, equal to STK2's store), G-S PASS on gate chunk 7, G-P: the probe priced the list with no label read.

**First stage on all 1,642 questions:** the fused list's ALL @100 55.2, @500 73.5, @1,000 81.7, @2,000 88.9, @2,500 **90.6**; the dense leg alone 48.2, 67.7, 76.9, 85.2, 88.1. The engine's own flat cells (`rrf1000_bN`) track it within 1.3 points; they fall BELOW it by **-1.3 [-2.0, -0.6]** at 1,530 and 1,586 nodes and by -0.9 [-1.7, -0.1] at 1,200 (no cause is tested here).

**Pick cells against the flat list at equal nodes** (ALL in percent with the Wilson 95 percent interval, ANY, mean distinct nodes, paired difference with its bootstrap interval; `*` = the flat reference ends at 2,500 nodes, shorter than the pool, so only the point difference is shown; the same eleven cells as section 14):

| cell | nodes | ALL | ANY | flat at equal nodes | difference | call |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rrf1000_b25+FULL:h1_c100+top25` | 50 | 49.1 [46.7, 51.5] | 91.1 | 48.0 | +1.1 [-0.2, +2.5] | LEVEL |
| `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 73 | 54.9 [52.5, 57.3] | 92.8 | 52.1 | +2.8 [+0.9, +5.0] | LIFTS |
| `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 152 | 64.5 [62.1, 66.8] | 97.1 | 59.4 | +5.1 [+3.7, +6.6] | LIFTS |
| `rrf1000_b400+FULL:h1_c100~s20+top800` | 635 | 77.5 [75.4, 79.4] | 98.8 | 76.2 | +1.3 [-0.1, +2.7] | LEVEL |
| `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 2,072 | 74.4 [72.2, 76.4] | 98.3 | 89.2 | -14.9 [-16.9, -12.6] | BELOW |
| `rrf1000_b400+STRUCT:h3_c25_v2000` | 1,984 | 77.6 [75.5, 79.5] | 98.9 | 88.9 | -11.3 [-12.9, -9.5] | BELOW |
| `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 2,448 | 87.1 [85.4, 88.7] | 99.7 | 90.5 | -3.3 [-4.6, -2.1] | BELOW |
| `rrf1000_b25+STRUCT:h3_c100_v5000` | 4,499 | 63.0 [60.7, 65.3] | 95.7 | 90.6 | -27.5 * | BELOW * |
| `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 5,415 | 91.0 [89.5, 92.3] | 99.8 | 90.6 | +0.4 * | LEVEL * |
| `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 5,867 | 91.1 [89.6, 92.4] | 99.8 | 90.6 | +0.5 * | LEVEL * |
| `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | 15,378 | 90.5 [89.0, 91.8] | 99.8 | 90.6 | -0.1 * | LEVEL * |

Of the 53 cells whose reference is not shorter than the pool, **4 LIFT, 21 LEVEL, 28 BELOW** (Prime, counted the same way: 13, 17, 23). **The lifts sit in small pools (46 to 220 nodes); from about 2,000 nodes up every cell is LEVEL or BELOW the fused list.** The best ALL within 5,000 nodes is 87.1 [85.4, 88.7] at 2,448 nodes, **3.3 [2.1, 4.6] below the flat list of the same size (90.5)**; no cell of the 63 reaches 95 at any size (the highest, 93.1 at 5,626 nodes, has a flat reference that ends at 2,500).

**FULL against STRUCT at equal walk and budget (the nine twins), in points:** -0.9 [-2.1, +0.4], +0.7 [+0.1, +1.3], +1.2 [+0.4, +2.0], +1.8 [+1.2, +2.6], **+2.0 [+1.2, +2.9]**, **+2.8 [+1.8, +3.8]**, **+5.5 [+4.4, +6.8]**, **+6.1 [+5.0, +7.3]**, **+6.9 [+5.5, +8.3]**: five of nine at 2 points or more (Prime: none). On Amazon the R and K edges are not neutral; **this read does not say which of R and K carries it** (K-only is 1,550,467 of 12,841,153 pairs, R-only 6,568,892).

**Dense leg alone against the fused list at equal size (flat cells `dense_topN`, no walk).** Every call is BELOW: -2.3 [-3.4, -1.1] at 5, -4.5 [-5.7, -3.3] at 20, -5.8 [-7.3, -4.5] at 50, **-7.1 [-8.7, -5.6] at 100**, -5.9 [-7.4, -4.4] at 200, -7.1 [-8.5, -5.7] at 400, -4.9 [-6.3, -3.5] at 1,000 (ALL 76.9 against the fused 81.7). `splade_top200` is 57.9 against the dense 56.3 and the fused 62.2. **Prime shows the same sign** (dense top 20 -3.6 [-4.6, -2.5], top 100 -4.8 [-5.9, -3.6], top 200 -4.0 [-4.9, -3.0], top 1,000 -3.1 [-4.1, -2.3]; `splade_top200` -7.0 [-8.0, -6.0]). These are first-stage lists, not pools: what the walk gives back is a different number (the declaration that follows, DRP1, reads it).

**Missed questions (section 15's read on the three cells, no choice made).** Cell A (2,448 nodes): **5 questions (0.3 percent) have no gold in the pool**, 206 have some but not all golds; ALL by gold count 1, 2, 3, 4 or more: 99.4, 95.3, 88.4, 76.9. Cell B (5,867): 4 and 142; cell C (15,378): 3 and 153. Where Prime's no-gold questions were single-gold questions with a gold outside the first list, **Amazon's pool misses are almost all partial questions** (206 of 211 at cell A: some golds in, one or more short), and the ALL rate falls with the gold count (99.4 to 76.9). X1, X2, X3 of section 15 all read true (ALL falls with the gold count; half or more of C's no-gold questions have a gold outside the 2,500 list, 3 of 3; at A, 3 of 5 no-gold questions are single-gold).

**Missed golds by degree and type (section 17's read, no expectation beyond G1 to G3).** 484 of 1,642 questions are single-gold; the single-gold golds missed are 3 (A), 2 (B), 2 (C), all `product`. G1 HELD (none of the 3 has union degree 0). G2 NOT HELD as worded (one node type among the single-gold golds). **G3 NOT HELD:** the missed golds' median union degree is 30.0 against 19.0 for the hits at A (33.5 at B and C): as on Prime, the misses are better linked than the hits (on 2 to 3 missed golds, so a direction only).

**Calls on section 19's expectations (Amazon; MAG's calls wait for its rows).**

- **A1 HELD on Amazon** (G-F, G-L, G-S, G-P pass with the same code, `zstk3.py` a10d8e0fb8f59432).
- **A2 HELD on Amazon.** Within 3,500 nodes the best LIFT is +5.1 [+3.7, +6.6] at 152 nodes (ALL 64.5), under the 6-point ceiling; it is a small-pool lift (see above).
- **A3 NOT HELD.** The expectation was a cell within 5,000 nodes at ALL 95.0 or more; the best is 87.1 at 2,448 nodes and no cell of the 63 reaches 95.
- **A4: no expectation on Amazon;** the nine twins are read as information above (five of nine at 2 points or more).
- **A5 HELD on Amazon.** 15.9 seconds for the first question (warm-up) and 1.1 for the second against the 30 allowed; peak 1.59 GB against 16.
- **A6 NOT HELD on Amazon.** The engine's flat cells fall below the fused list by up to 1.3 [0.6, 2.0] points at equal nodes (the bound was 1 point; Prime 0.5 and 0.8).

**What this says about the user's target (95 percent of the golds at about 2,500 nodes, one rule on all ten).** On Amazon the fused first-stage list already holds 90.6 percent of the golds at 2,500 nodes, and **no walk beats it at that size** (the best cell is 3.3 points below, 87.1 against 90.5); the walk gives back only in small pools. Prime was the opposite (walk +20.9 at equal nodes). So the transfer of the seven's rule is **not universal across the two STaRK KBs read so far**: it LIFTS Prime and does not lift Amazon at the target size. Nothing is chosen or adopted; the best cell is read post-hoc; no rule, cell or link set changes from this page.

**Not read:** MAG (its rows wait for its links row); which of R and K carries the FULL-over-STRUCT gain on Amazon; what the walk returns when the SPLADE leg or K is removed (DRP1).
