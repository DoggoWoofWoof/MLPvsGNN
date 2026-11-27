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
