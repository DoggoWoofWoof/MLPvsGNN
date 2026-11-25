# PL6: the coverage target, round 5 (one selection rule on all seven, no node bound, a budget ladder that grows until the target is met)

Declared 25 November 2026, before any PL6 coverage number exists. What I have read: PL4's FILED records and page (section 14 re-filed with all seven datasets at 18cccdc1),
PL5's declaration (3168891e) and the PL5 records that were on disk when this page was written (the dev selects of MultiHop-RAG, SQuAD and MetaQA, the frozen confirmations of MultiHop-RAG and
MetaQA, MetaQA's D1/D2, the G-R files of the five laptop datasets: `outputs/pl5/<d>/select.json`, `confirm_frozen.json`, `miss.json`, `outputs/pl5/gr_*.json`), the dev tables of PL4's
selects for all seven datasets (they hold the cells above the bound), and the code of `zpl2g.py` to `zpl5g.py`. **PL5's MuSiQue, WebQSP, HotpotQA and 2Wiki selects and confirmations are not read
and not needed to declare this page; PL5 is not changed by it.** Before this commit I ran only label-free things: the cell list and its sha256 (`zpl6g.py cells`), the cost of the ladder's walks
on MuSiQue (`zpl6g.py time`: seconds and nodes scored, 2 dev questions per rung, no label read), the self-test (`selftest.json`: PASS in 68.6 s; agreement counts, no coverage), and a code test of `select`
and `call` on MultiHop-RAG, MetaQA and SQuAD **into a scratch folder** (the rule reads PL5's already-filed tables; the output was not kept and is not part of the record). **Nothing here trains, no
model is fitted, no LLM or spaCy is used, and every pool function is gold-free.** The pools of record and PL2's to PL5's code and records are untouched. This page is the fifth round of the coverage target
(docs/PL1_POOLS_ON_UL2_LAYOUT.md section 7).

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how much ever is possible".
- 25 Nov, about 15:00, in chat, asked whether the 2,500 mean-node bound of PL2 to PL5 may be lifted for the three datasets that stay short under it: **"i do not mind just rules should be universal."**
  This page is the reading of that ruling: the bound is lifted only if ONE selection rule, the same on all seven datasets, replaces it.
- Earlier, still binding: "the pools or the rules or links all should be universal"; "i dont want to freeze pools for the sake of it, make sure we freeze when we know it is the best for that specific arm";
  pools first and parameter-free (no learning, no model, no LLM, no spaCy); one model per family; the hub line is CLOSED (UL2 8.15): nothing below reads a node's degree or adds a hub step.
  Every rule is universal on all seven datasets (MultiHop-RAG, MuSiQue, SQuAD, MetaQA, WebQSP, HotpotQA, 2Wiki), carried everywhere or nowhere.

## 1. What the records say, and what I had seen above the bound before writing this page

The bound is lifted after I had seen what the cells above it reach. That is stated here so the page cannot be read as a blind test of the lifting. Dev, frozen lists, over PL4's 4,206 cells (PL5's 6,078
add cells that the same reading covers: MetaQA, MultiHop-RAG and SQuAD were read on them and give the same picks; MuSiQue, WebQSP, HotpotQA and 2Wiki are not read on them):

| dataset | smallest cell at 0.95 inside 2,500 nodes (dev) | smallest cell at 0.95 at any size, or the highest ALL if none (dev) |
|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | the same cell |
| squad | `rrf1000_b15` 0.9519 at 15.0 | the same cell |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | the same cell |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | the same cell |
| musique | none: highest ALL 0.8781 at 2,298.7 (SHORT) | none: highest ALL **0.9355 at 5,783.4** (`rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr`) (SHORT) |
| metaqa | none: highest ALL 0.8978 at 2,339.7 (SHORT) | **`rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6** (reaches 0.95; eval, frozen, filed in PL5: **0.9582 [0.9541, 0.9620] at 4,972.8** against 0.8968 at 2,335.9 for the bounded pick) |
| webqsp | none: highest ALL 0.7731 at 2,442.4 (SHORT) | none: highest ALL **0.8540 at 6,021.1** (`rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq`) (SHORT) |

Three facts follow, and the design rests on them.
1. **The bound binds on three datasets only (MuSiQue, MetaQA, WebQSP).** On the other four the smallest cell that reaches 0.95 is inside the bound, so any rule of the form "the smallest cell that reaches 0.95" picks the same cell with or without it.
2. **MetaQA's reading above the bound is seen** (dev 0.9613 at 4,983.6; eval 0.9582 at 4,972.8). The call below will read ABOVE on MetaQA for that reason. That is the answer to the user's question for that dataset, not a blind result.
3. **MuSiQue (0.9355) and WebQSP (0.8540) stay short of 0.95 even at 5,800 to 6,000 nodes**, and every one of their best cells is a walk whose visited budget is 5,000. Whether a larger budget keeps raising ALL is the open
   question; it is what the ladder (section 4) measures. D2 (PL4 14.5) says WebQSP's shortfall at the 2,500 bound is the per-seed and per-frontier caps and MetaQA's the hop-3 budget; it was not computed for the above-bound cells.

## 2. What is fixed (PL2's to PL5's pages, by reference)

Everything of PL2 sections 2 to 4, PL3 sections 2 and 4, PL4 sections 2, 4 and 6 and PL5 section 4 (the walks, `cs`, `vr`, `sp`, `rk`, `fu`, `rr`, `eq`, `s10`, `s20`, the budget-following walk `bf`, the bases, the sizes, the measures, the statistics) is
unchanged and read by the same code (`zpl2g.py` to `zpl5g.py`, imported, not edited): the substrate, the dev carves and eval carves, the frozen first stage for selection and for confirmation (e3v beside), the seeds, ALL = `all_gold_at_pool`,
mean nodes = `candidates_mean`, Wilson 95% on ALL, the paired bootstrap of 1,000 resamples, the floors 0.95 and 0.98. **The 6,078 cells of PL2, PL3, PL4 and PL5 are this page's rung 0** and are read from their filed dev grids; PL6 recomputes none of them for its reading.
The one thing this page removes is the 2,500 bound in the selection rule and in the confirmation (it stays as a flag).

## 3. What this round tests, and what it does not

| name | here |
|---|---|
| one selection rule on all seven, no bound (section 5) | rule U: the smallest mean nodes with dev ALL >= 0.95, else the highest dev ALL; price filed beside every call |
| a budget ladder (section 4) | the visited budget of the walks that held the dev frontier above the bound is raised in four steps, 10,000, 20,000, 40,000 and 80,000, for a dataset only while rule U is SHORT and the best ALL is still rising; no other change to any walk |
| a new walk, order, window or key, the weights of NER and KNN rows, a parent-conditioned rank, label-ordered caps | **not in this round** (PL5 section 12 names them; each is declared on its own page) |

## 4. The ladder, exactly

**Settings.** For r = 1..4 and V = 10,000 x 2^(r-1) (10,000, 20,000, 40,000, 80,000): `h2_c100_v{V}` and `h3_c100_v{V}` (PL2's `h2_c100_v5000` and `h3_c100_v5000` with the visited cap V: per-seed cap 100, per-frontier cap 100) and
`h2_bf_v{V}` and `h3_bf_v{V}` (PL5's budget-following walk with the visited budget V and the 512 window). `zpl6g.py` adds these eight settings per rung to PL5's table and teaches PL5's walk parser their names **after** the earlier
rounds' code-path hashes are computed and cached (so every earlier grid head stays valid); gate G-P checks the pinned files' hashes. The expansion, the orders, the grading are PL3's, PL4's and PL5's code unchanged.

**Walks (13 per rung, regime:setting~mods, the mods in PL5's order).** Written for rung r with its V:
`STRUCT:h3_c100_v{V}`, `STRUCT:h3_c100_v{V}~cs-sp-eq`, `FULL:h3_c100_v{V}~cs-sp-eq`, `FULL:h3_c100_v{V}~cs-sp`, `FULL:h3_c100_v{V}~cs-vr-eq`, `FULL:h2_c100_v{V}~cs-vr`, `FULL:h2_c100_v{V}~s10-cs-vr`, `FULL:h2_c100_v{V}~s20-cs-vr`,
`FULL:h2_c100_v{V}~s20-cs-rk`, and the budget-following `FULL:h3_bf_v{V}~cs-vr`, `FULL:h3_bf_v{V}~cs-fu`, `STRUCT:h3_bf_v{V}~cs-fu`, `FULL:h2_bf_v{V}~s10-cs-vr`.
**Why these 13, said so that nobody reads them as neutral.** The nine c100 walks are the walks of the highest-ALL cells above the bound in PL4's reads (the dev tables I had seen): MetaQA's `STRUCT:h3_c100_v5000`, WebQSP's `h3 ~cs-sp` and `~cs-sp-eq`
(FULL and STRUCT), MuSiQue's `FULL:h2 ~s20-cs-vr`, `~s10-cs-vr` and the plain `~cs-vr`, and `FULL:h3 ~cs-vr-eq`, and 2Wiki's `FULL:h2 ~s20-cs-rk`. The four budget-following walks are the PL5 walks with the same budgets, chosen **without** any PL5 number
for MuSiQue or WebQSP (not read); they are a hedge, not a finding. A walk that is not on this list is not tried on the ladder; if the ladder ends short, which walk to add is a later page's question.

**Cells (per rung).** Each walk x 3 bases (`rrf1000_b25`, `rrf1000_b400`, `rrf1000_b1600`: the smallest, a middle and the largest of PL2's first-stage budgets) as a union cell, and as a ranked cell `+top{M}` with M = V x f / 4 for f = 1, 2, 3
(2,500, 5,000, 7,500 at rung 1; 5,000, 10,000, 15,000 at rung 2; 10,000, 20,000, 30,000 at rung 3; 20,000, 40,000, 60,000 at rung 4): 13 x 3 x (1 + 3) = **156 cells per rung, 624 in all, none repeating a PL2 to PL5 cell**.
Cell list sha256 (`P2.cells_sha256`, the sorted names), per rung: rung 1 `1dd177c06a0d5c8895dd0cddb4e1001d45652fd747d9b4e401241b471d817373`, rung 2 `0b296eb8a030caef3ee13452a9ae0b1a7e679095b1068fbd3550f27fe66b9bbf`,
rung 3 `f3afe89b1f3e1da581cb8fae83b657bedfc61e4cc4fe0544cb44dc3eb5499644`, rung 4 `1d0d4c56aa6aa194b4d2899f7f3b76004ed9737340fc7a633331fffafe8d8900`; all 624: `6cc6d826ff78b74b7fd94796c62b0d31105f939ddfb71106413c9232a8d4607d` (`outputs/pl6/cells.json`).

**When a rung runs (the same rule on every dataset).** Let best(k) be the highest dev ALL over all cells of rungs 0..k (no bound). The state of a dataset is, in this order:
- **REACHED** if best(k) >= 0.95 (rule U picks a cell that reaches the target; no further rung);
- **STOPPED_RISING** if k >= 1 and best(k) - best(k-1) < 0.005 (the ladder stops: ALL has stopped rising);
- **GRAPH_END** if k >= 1 and V(k) is at least the number of nodes of the dataset's graph (a larger visited budget cannot add nodes);
- **LADDER_END** if k = 4 (rung 5 would be declared on its own page);
- otherwise **RUN_RUNG**: rung k+1 runs (dev grid, frozen lists).
Rung 0 is always read first, so a dataset that reaches 0.95 inside PL2 to PL5's cells runs **no rung**. On today's records that is MultiHop-RAG, SQuAD, HotpotQA, 2Wiki and MetaQA; MuSiQue and WebQSP ask for rung 1.
`zpl6g.py ladder` prints each dataset's state from the filed records; `grid` refuses a rung the rule does not ask for.

## 5. The rule, the confirmation and the call

**Selection, rule U** (dev, frozen lists, per dataset, over rung 0's 6,078 cells and the ladder cells of the rungs read): **U95** = the smallest mean nodes with dev ALL >= 0.95 (ties to the higher ALL, then the name); if no cell reaches it, the highest ALL
(ties to fewer nodes, then the name), and the dataset is SHORT. **U98** likewise at 0.98, beside. **There is no node bound anywhere in it.** Readings (each by the same rule): `all` (the decider), `rung0` (the 6,078 cells), `bounded` (the 6,078 cells inside 2,500 nodes: the same rule on the same cells as PL5's own `R95_all`, so it names PL5's pick; both are filed, and a difference would show in the record). The whole curve and the frontier of each reading are filed.

**Confirmation** (eval, nothing re-chosen, frozen lists; e3v beside): the cells U95 and U98 name in the readings `all` and `rung0`, PL5's bounded pick (the incumbent), the matched-node-count picks at N = 3,000, 4,000, 5,000, 6,000, 7,000, 10,000, 15,000, 20,000, 30,000, 40,000, 60,000, 80,000
in the readings `all` and `rung0`, and PL2's filed own construction. Cells PL5's confirmation grid already holds are read from it, not recomputed (`zpl6g.py confirm` builds only the others).

**The call.** On eval, frozen lists, per dataset: NEW = U95 of the reading `all`, OLD = PL5's bounded pick (R95 of PL5's reading `all` inside 2,500: what PL2 to PL5 carried). Paired bootstrap of the ALL difference (NEW - OLD) and the ratio of mean nodes; one generator per dataset,
`numpy.random.default_rng([20261125, k])`, k the dataset's place in the order MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MuSiQue, MetaQA, WebQSP; the status is PL3's `status_of` unchanged, clause by clause:
- **BELOW**: NEW misses 0.95 where OLD reaches it, or the ALL difference's upper end is below 0 with the nodes ratio's lower end at or above 1.
- **ABOVE**: NEW reaches 0.95 where OLD does not, or the nodes ratio's upper end is below 1 with ALL not BELOW, or the ALL difference's lower end is above 0 with the nodes ratio's upper end at or below 1.
- **LEVEL**: anything else.
Beside the status, filed in the same record and deciding nothing: the **ALL-only status** (ABOVE if the ALL difference's lower end is above 0, BELOW if its upper end is below 0, else LEVEL: it does not credit or charge nodes, and it is the status that says whether the extra nodes bought coverage).
**CARRY_PL6** when all seven are read, **the universality check holds**, no dataset is BELOW, and at least one is ABOVE; INCOMPLETE while fewer than seven are read; otherwise NOT_CARRIED (PL5's bounded rule stays what PL2 to PL5 used). **The universality check:** on every dataset whose
bounded pick reached 0.95 on dev, U95 is that same cell (the smallest cell that reaches the target inside the bound is the smallest that reaches it anywhere, so this holds by construction; the check is that the code agrees: it is recorded per dataset as `universal_same_as_bounded`, and CARRY_PL6 is not possible while any of them is false, which could only be a code error).
On today's records that is MultiHop-RAG, SQuAD, HotpotQA and 2Wiki: they come out unchanged, with an identical eval record, status LEVEL.

**The price, filed beside every call (it decides nothing, and it is the number the user weighs).** For every dataset the call file holds: mean nodes on eval for NEW and for OLD (their difference is the extra nodes per question), NEW's 95th percentile and maximum nodes per question, the ratio NEW/OLD with its interval, and the graph's node count,
so the cost of the lifted bound is read as nodes per question and as a share of the graph. It also holds beside: rule U's U95 pick over `all` against rule U's pick over `rung0` alone (what the ladder bought over rung 0 under the same rule), U98 the same way at 0.98,
and the matched-node-count picks of `all` against `rung0` at the twelve budgets above (the frontier reading above the bound; per budget the paired ALL interval and the ALL-only status; a dataset is BELOW if any budget is BELOW, ABOVE if at least two are ABOVE and none BELOW, else LEVEL).

## 6. What the ladder costs, and how MetaQA reads

**Measured on MuSiQue before this commit** (`zpl6g.py time`, 2 dev questions, no label read; the first rung includes cold caches; MuSiQue's graph has 117,534 nodes; peak 0.62 to 0.64 GB; files `outputs/pl6/time_musique_r{1..4}.json`):

| rung | V | seconds per question (156 cells, 13 walks) | nodes scored per question |
|---|---|---|---|
| 1 | 10,000 | 10.1 | 87,985 |
| 2 | 20,000 | 15.0 | 93,232 |
| 3 | 40,000 | 14.8 | 93,476 |
| 4 | 80,000 | 15.4 | 93,476 |

Read: **rungs 3 and 4 score exactly as many nodes as each other and 0.3 percent more than rung 2**: on these two questions the walks stop growing around V = 20,000 (the caps and the graph's reach end them, not the budget). If that holds on the dev carve, rung 3 adds almost nothing to rung 2, so the
stop rule (ALL has stopped rising, a gain under 0.005) ends the ladder at rung 3 at the latest and rung 4 never runs; if rung 2 itself gains under 0.005 over rung 1, rung 3 does not run either. **MuSiQue cost, laptop or host (the same CPU seconds; 1,534 dev questions):** rung 1 about 15,400 s of CPU summed over the parts, rung 2 about 23,000 s, rung 3 or 4 about 23,000 s each (worst case, all four rungs: about 85,000 s, 23.5 hours of CPU; with three parts in
parallel about 8 hours of wall time; the expected case, rungs 1 and 2 only: about 38,000 s CPU, about 3.5 hours of wall time with three parts). Two questions are a small sample; the real numbers are in the grid heads. **WebQSP (1,507 dev questions): not measured before this commit** (the laptop's memory is taken by PL5's WebQSP grid). An estimate, good to a factor of two, from PL5's G-R on the same cells (WebQSP 1,763 s against MuSiQue 1,283 s for the same 400 + 150 questions of cells, a ratio of 1.4):
about 14 s per question at rung 1 and 21 s at rungs 2 to 4, so about 21,000 s of CPU at rung 1, 31,600 s at each later rung, 116,000 s (32 hours of CPU, about 11 hours of wall time with three parts) if all four rungs run, and about 52,700 s (14.6 hours of CPU, about 5 hours of wall time) if the ladder stops after rung 2. The measured probe
(`zpl6g.py time --ds webqsp --rung 1 --n 2 --write`, 2 dev questions, no label read, decides nothing) is filed beside the results as `outputs/pl6/time_webqsp_r1.json` when the laptop's memory allows; if it differs from the estimate, the page is not changed and the probe is the record.
**Memory:** the probe's peak is 0.64 GB on MuSiQue; the ladder's arrays are the same size as PL5's per question (scores for at most about 100,000 nodes in float64, under 1 MB per walk); WebQSP's own peak was 5 to 6 GB per part in PL4's and PL5's grids, and its rows ask for 8 GB.
**Host (where the ladder runs: host first):** the same CPU seconds on 2-cpu rows (the host's speed is not assumed faster): per rung and dataset three grid parts (`pl6-<ds>r<rung>g0` to `g2`, 2 cpus; 3 GB MuSiQue, 8 GB WebQSP) and a select row (`pl6-<ds>r<rung>sel`, 1 cpu, 3 GB),
plus the host's G-R on MuSiQue (`pl6-musiquegr`, 2 cpus, 3 GB; about 1,300 s on the laptop) that the grid rows wait on. Rung 1 is 9 rows (6 grid parts, 2 selects, the G-R); with MuSiQue's numbers above one rung of MuSiQue is 15,400 s to 23,600 s of CPU in 3 parts (4.3 to 6.6 CPU-hours), one rung of WebQSP is
(estimate) 5.8 to 8.8 CPU-hours. `zpl6g.py rows --rung R --ds ...` prints the rows for the datasets the rule asks for; rung 1's are in `outputs/pl6/feed_rows.txt` and are **conditional**: they need PL5's MuSiQue and WebQSP selects (`outputs/pl5/<d>/select.json`)
and PL5's MuSiQue dev grids (for the G-R) pushed as inputs, none of which exists yet, and the grid refuses to run (and its row fails without a number) if the rule does not ask for the rung. **Rung 2 and later rows are not in the file:** whether they run is decided by the previous rung's select, and I print them then.
**The laptop** reads the other half: `select --rung 0` on all seven (a table read, seconds), `confirm` of the picks (MuSiQue 2,417 and WebQSP 1,503 eval questions, one pick-set grid each; time not estimated, memory permitting) and `call`; the laptop is also the fallback for a rung the host cannot take (the same commands without `--host`).
**No host row is needed for HotpotQA or 2Wiki:** they run no rung, and every cell their picks name is in PL5's confirmation grid.
**MetaQA** reads at rung 0, with no rung run: U95 = `rrf1000_b25+STRUCT:h3_c100_v5000`, dev 0.9613 at 4,983.6 nodes, eval (frozen, filed in PL5) 0.9582 [0.9541, 0.9620] at 4,972.8 against the bounded pick's 0.8968 at 2,335.9: **2.13 times the nodes** for **+6.1 points** of ALL, the target
reached; on e3v lists PL4 filed the bounded pick at 0.9296 at 2,325.6 and the unbounded cell is not yet read there (`zpl6g.py confirm --ds metaqa --first e3v` on the laptop after the select: a beside-read that decides nothing). That is a price of about 2,640 more nodes per question on every MetaQA question.

## 7. Stop rule

The 13 walks, the eight settings per rung, the bases, the sizes M, the ladder's constants (the target 0.95, the rise 0.005, V = 10,000 x 2^(r-1), at most four rungs), rule U, the readings, the budgets of the frontier reading above the bound, the calls and their statistics and the carves are fixed by this commit; every output records the sha256 of its
cell list and the code-path hash, `zpl6g.py` refuses a grid whose cell list is not the declared one and a select whose heads carry another code-path hash, and `call` refuses to file when two datasets differ (G-U). Nothing is added after a number is read. A dataset that ends short
(STOPPED_RISING or LADDER_END) is reported with D1 and D2 at its chosen cell (`zpl5r.py miss`, post-hoc, labels read, deciding nothing), and the next change is declared on its own page.

## 8. Gates (each is a failure that stops that dataset's reading; none of its numbers is read)

- **G-P (the pinned code).** The code-path hashes of PL2, PL3, PL4 and PL5 computed by this process, before the ladder's settings are added, equal the ones filed at PL5's self-test; `zpl5g.py`'s script hash equals the self-tested file's.
- **G-R (the earlier cells are reproduced with the ladder's settings added).** `zpl6g.py gr` builds PL5's 1,872 cells on the first 100 dev questions and PL4's 1,932 on the first 40 and compares every present-count and pool size to the filed dev arrays element for element.
  Required on every dataset that runs a rung (laptop: its own; host: the host's G-R on MuSiQue, as PL5's).
- **Engine self-test (PASSED before this commit).** Cell lists and sha256s, every cell name parses, the 624 cells overlap no earlier cell; the engine's 13 walks at rung 1 against PL5's per-node reference on 1 real question each of MultiHop-RAG, SQuAD and MuSiQue:
  13 checks each, 0 mismatches (`outputs/pl6/selftest.json`). The larger rungs differ from rung 1 only by V.
- **G-F, G-D.** No pool function is new; the earlier rounds' G-F (gold-free) and G-D (determinism) hold for the code this page imports, which G-P pins.
- **G-U (universal).** One code-path sha256 (the ladder code and constants and PL5's hash) in every grid head; `select` and `call` refuse a head that differs.

## 9. What the ladder can and cannot show

It can show whether a visited budget of 10,000 to 80,000 raises dev ALL on MuSiQue and WebQSP past the 0.9355 and 0.8540 that V = 5,000 reached, and at what node count. It cannot reach a gold the first stage and the graph together do not reach (PL4 14.5, 300 dev questions: 2 golds on MetaQA, none on MuSiQue
and WebQSP, that the graph cannot reach from the first stage), a gold cut by a cap of 100 per node (the c100 walks keep the cap; the bf walks have none but the window), or a gold a different order would place first. It adds no semantic signal. If the c100 walks saturate (the probe suggests the walks stop growing near 20,000
nodes on MuSiQue), the ladder ends by STOPPED_RISING and the answer is that the visited budget was not the limit.

## 10. Code, rows, outputs

- **New module:** `outputs/mp_unified/zpl6g.py` (commands `cells`, `time`, `gr`, `ladder`, `grid`, `select`, `confirm`, `call`, `rows`, `selftest`; every command except `cells`, `ladder`, `call` and `rows` takes `--host`), committed with this page. It imports `zpl5g.py` and edits none of the earlier modules.
  The diagnostics at the end use `zpl5r.py` (`miss`, `report`), which is not part of any hash.
- **Outputs:** `outputs/pl6/<d>/{grid_dev_frozen_r<rung>*.npz (arrays, uncommitted), grid_dev_frozen_r<rung>*.json (heads), select_r<rung>.json, select.json, confirm_frozen.json, grid_eval_frozen_confirm.*}`, `outputs/pl6/{cells.json, selftest.json, gr_<d>.json, call.json, time_*.json, feed_rows.txt}`.
- **Host first, laptop for the rest** (one heavy laptop job at a time, three parts of a dev grid in parallel): the ladder's dev grids for MuSiQue and WebQSP are host rows (section 6) once PL5's selects of those two are in; the laptop reads `select --rung 0`, the frozen confirmation of the picks (MuSiQue 2,417 eval questions, WebQSP 1,503) and `call` over the seven, and takes a rung itself only if the host cannot.

## 11. What follows

1. PL5's reads finish first (its page is not changed). Then `zpl6g.py select --rung 0` on all seven (rule U over rung 0's 6,078 cells), `ladder` to see which datasets ask for a rung, the rungs, `confirm`, `call`, and section 12 of this page with the tables:
   per dataset NEW against the bounded pick with ALL, nodes and the price, the ladder's state, the ALL-only status, the frontier reading above the bound, what the ladder bought, what is still short, D1/D2.
2. If CARRY_PL6, the user decides whether the pools downstream (PL1's rebuild) take the unbounded cells; this page decides nothing for the downstream stages' memory or time.

## 12. Disclosed

- **The bound was lifted after the beside reads above 2,500 were seen** (section 1): PL4's any-size tables for all seven datasets and PL5's for MultiHop-RAG, SQuAD and MetaQA, including MetaQA's eval reading at 4,972.8 nodes. I proposed to lift it only under one rule on all seven because the user ruled "rules should be universal";
  the alternative of a different single bound (7,000 say) is also universal and would leave 2Wiki and HotpotQA unchanged too; I did not choose it because any fixed number is arbitrary where the target (0.95 at the fewest nodes) is not.
- **Selection is on the dev carve and the confirmation on eval**, as PL2 to PL5: "the smallest cell whose ALL reaches 0.95" is read on dev (1.5k questions; 224 for MultiHop-RAG) and then re-read on eval; the winner's dev ALL is optimistic (about 6,700 cells are selected over), so the eval read decides.
  A dataset whose dev pick reaches 0.95 may read below it on eval, and the call says so.
- **The universality check is an identity** for the four datasets that reach the target inside the bound; it can fail only by a code error, and it is filed as such, not as a finding. The substantive universality is that one rule, the ladder's stop rule, and one set of walks apply to all seven.
- **The 13 walks are partly chosen from PL4's above-bound tables that I had seen** (section 4); the four budget-following walks are not chosen from any PL5 number of the short datasets.
- **The ladder's cost sample is two questions per rung on one dataset** (MuSiQue); WebQSP's is stated in section 6 as measured or as not measured.
- **A pool of 5,000 to 6,000 nodes per question is large** for the stages that read it (a re-ranker or a graph model's forward pass over the pool); the price is filed beside every call so that the user can weigh it. The price clause of the old rule (the nodes ratio) can credit a gain bought with fewer nodes at an unchanged ALL, as in PL4's MuSiQue,
  and cannot say that 2.13 times the nodes is worth 6.1 points: the ALL-only status and the price columns say what was bought.
- The six datasets' graph is the old package graph (as PL2 to PL5 disclosed); MultiHop-RAG reads its B0 layout. The UL2 layout is not materialized on the six.

## 13. Results (26 Nov 2026; 7 of 7 datasets read)

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl6/<d>/{select,select_r<k>,confirm_frozen,miss}.json`, `outputs/pl6/call.json`, `outputs/pl6/gr_*.json`, the grid heads) by
`outputs/mp_unified/zpl6g.py` (`select`, `ladder`, `confirm`, `call`) and `zpl5r.py miss`. PL5's pages and records are not changed; PL5's selects are this page's rung 0. Nothing was changed in the rule, the cells or the engine after a number was read.
Section 12 of this page is the declaration's own disclosures; this results section is therefore 13 (section 11 of the declaration says 12).

**How to read this page (the rulings that came after the declaration).** The user's ruling of 25 Nov about 15:00 lifted the 2,500 bound only under one rule on all seven (section 0); he then said (relayed by the coordinator, 25 Nov, before PL7) "by lifting that bound i asked you to do analysis, we still need to start restricting it and bring towards 2.5k". **This page is therefore the analysis he asked for: the map of what reaching 0.95 costs when nothing bounds the pool.** The call below carries its rule's name, but no cell above 2,500 nodes is taken downstream from this page: the restriction is PL7 (`docs/PL7_COVERAGE_RESTRICTION.md`, 24f4892e, declared before any number, one rule on all seven, on the current links, provisional until the link set is final), which reads the carry at 2,500 only. Nothing of PL2 to PL5 is changed by this page.

### 13.1 Gates and cost

The engine `zpl6g.py` is the declaration commit's (744811e6): script sha256 starts `8a595176ed61`, the code-path hash `cce8c3fb01e4` is equal in every select head and `call.json` (`code_path_equal_everywhere` true; G-U PASS). **G-P** and the engine self-test PASSED before the commit (`selftest.json`). **G-R** (PL5's 1,872 cells on the first 100 dev questions and PL4's 1,932 on the first 40, rebuilt by this engine with the ladder's settings added, compared element for element) PASSED with 0 present and 0 size mismatches on: laptop none; host (MuSiQue's, read for every host rung) MuSiQue. Cost of the ladder's dev grids (sum of the three parts): MuSiQue rung 1 8,178 s, 0.63 GB peak (host); MuSiQue rung 2 12,755 s, 0.63 GB peak (host); WebQSP rung 1 4,929 s, 7.67 GB peak (host); WebQSP rung 2 6,555 s, 7.72 GB peak (host); WebQSP rung 3 8,869 s, 7.74 GB peak (host). The two-question probes filed before the runs: MuSiQue rung 1: 10.1 s/q, 87,984 nodes scored; MuSiQue rung 2: 15.0 s/q, 93,232 nodes scored; MuSiQue rung 3: 14.8 s/q, 93,476 nodes scored; MuSiQue rung 4: 15.4 s/q, 93,476 nodes scored; WebQSP rung 1: 18.1 s/q, 167,900 nodes scored; WebQSP rung 2: 14.7 s/q, 245,484 nodes scored; WebQSP rung 3: 14.1 s/q, 294,674 nodes scored; WebQSP rung 4: 9.5 s/q, 294,674 nodes scored.

### 13.2 The ladder: what the rule asked for, per dataset

The stop rule reads `best[r]` = the best dev ALL over rungs 0 to r: REACHED when it is 0.95 or more, STOPPED_RISING when the last rung raised it by less than 0.005, GRAPH_END when V has met the graph, LADDER_END after rung 4, otherwise the next rung is asked for.

| dataset | ladder state | rungs read | best dev ALL, rung 0 -> last rung | U95 of all cells (dev ALL at dev nodes) | PL5's bounded pick (dev ALL at dev nodes) | graph nodes |
|---|---|---|---|---|---|---|
| MultiHop-RAG | REACHED | 0 | 1.0000 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | 609 |
| SQuAD | REACHED | 0 | 1.0000 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | 20,233 |
| HotpotQA | REACHED | 0 | 0.9980 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | not filed here |
| 2Wiki | REACHED | 0 | 0.9786 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | not filed here |
| MuSiQue | REACHED | 2 | 0.9355 -> 0.9485 -> 0.9668 | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` 0.9550 at 15,400.0 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8898 at 2,464.3 (SHORT) | 117,534 |
| MetaQA | REACHED | 0 | 0.9653 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 (SHORT) | not filed here |
| WebQSP | REACHED | 3 | 0.9257 -> 0.9416 -> 0.9496 -> 0.9569 | `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` 0.9555 at 39,571.6 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8759 at 2,094.2 (SHORT) | 2,592,894 |

- **MuSiQue** asked for two rungs. Rung 0 (PL5's cells) reaches 0.9355 on dev at best; rung 1 (visited budget V = 10,000) 0.9485, a rise of 0.0130 and still under 0.95; rung 2 (V = 20,000) 0.9668: REACHED, so rungs 3 and 4 were not run. U95 is the smallest cell at 0.95 or more on dev: a hard cut of the V = 20,000 budget-following walk, `+top15000` (dev 0.9550 at 15,400.0 nodes; a pool never exceeds the base's 400 plus the 15,000 the cut keeps).
- **WebQSP** asked for three rungs: 0.9257 (rung 0), 0.9416 (V = 10,000), 0.9496 (V = 20,000, a rise of 0.0080 and still under 0.95), 0.9569 (V = 40,000): REACHED at rung 3, so rung 4 was not run. U95 is the V = 40,000 walk with base 400 and no cut (dev 0.9555 at 39,571.6 nodes, 1.5% of the graph's 2,592,894).
- **MetaQA** reaches 0.95 on dev at rung 0 above the bound (U95 `rrf1000_b25+STRUCT:h3_c100_v5000`, dev 0.9613 at 4,983.6 nodes): no rung was run for it, as declared in section 6.
- **MultiHop-RAG, SQuAD, HotpotQA and 2Wiki** reach the target inside the bound at rung 0; their U95 is PL5's bounded pick (the universality check, below). **No dataset ended short on dev (every ladder state is REACHED), so section 7 of the declaration asks for no D1 or D2 on this page** (13.6).


### 13.3 The call (read 7 of 7: CARRY_PL6)

NEW = U95 of the reading `all` (rule U, no node bound), OLD = PL5's bounded pick (R95 of PL5's reading `all` inside 2,500, what PL2 to PL5 carried); eval, frozen lists, paired bootstrap, one generator per dataset. Clauses as section 5:
BELOW (1) NEW misses where OLD reaches / (2) ALL difference upper end < 0 with ratio lower end >= 1; ABOVE (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (not BELOW) / (3) ALL difference lower end > 0 with ratio upper end <= 1. The ALL-only status ignores nodes.

| dataset | NEW (U95, eval ALL at mean nodes) | OLD (PL5 bounded pick, eval ALL at mean nodes) | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | status | ALL-only status | BELOW 1 / 2; ABOVE 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | **LEVEL** (the same cell) | LEVEL | no / no; no / no / no |
| SQuAD | `rrf1000_b15` 0.9554 at 15.0 | `rrf1000_b15` 0.9554 at 15.0 | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | **LEVEL** (the same cell) | LEVEL | no / no; no / no / no |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | **LEVEL** (the same cell) | LEVEL | no / no; no / no / no |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | **LEVEL** (the same cell) | LEVEL | no / no; no / no / no |
| MuSiQue | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` 0.9640 at 15,400.0 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8846 at 2,438.2 | yes / no | +7.94 [+6.91, +8.98] | 6.3160 [6.3025, 6.3303] | **ABOVE** | ABOVE | no / no; yes / no / no |
| MetaQA | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9582 at 4,972.8 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | yes / no | +6.14 [+5.68, +6.64] | 2.1289 [2.1280, 2.1298] | **ABOVE** | ABOVE | no / no; yes / no / no |
| WebQSP | `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` 0.9474 at 39,529.5 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8789 at 2,094.6 | no / no | +6.85 [+5.59, +8.12] | 18.8722 [18.7922, 18.9507] | **LEVEL** | ABOVE | no / no; no / no / no |

- **BELOW: 0.** ABOVE: 2 (MuSiQue, MetaQA); LEVEL: 5 of the 7 read.
- **The universality check** (on every dataset whose bounded pick reached 0.95 on dev, U95 is that same cell): datasets 2Wiki, HotpotQA, MultiHop-RAG, SQuAD; unchanged on all of them: **yes** (an identity by construction, recorded as a code check).
- **The call: CARRY_PL6** (BELOW 0, ABOVE 2, LEVEL 5; read 7 of 7). CARRY_PL6 needs all seven read, the universality check, no BELOW, and at least one ABOVE.

- **What decides each status.** MuSiQue and MetaQA are ABOVE through clause 1: the new pick reaches 0.95 on eval (MuSiQue 0.9640, MetaQA 0.9582) where the bounded pick does not (0.8846, 0.8968). **WebQSP is LEVEL**: its new pick does not reach 0.95 on eval either (0.9474, short by 0.26 points, where it reached 0.9555 on dev), so clause 1 does not fire, and the ALL difference's lower end is above 0 (+6.85 [+5.59, +8.12] points) but the nodes ratio is 18.87, so clause 3 does not either. The ALL-only status says ABOVE: the 37,435 extra nodes per question bought coverage, and not enough of it.
- **The price in nodes.** The three datasets whose pick moved use 2.13 times (MetaQA), 6.32 times (MuSiQue) and 18.87 times (WebQSP) the nodes of the bounded pick, with 95th percentiles and maxima within 2.1% of the means (the hard cut and the budget-following walks make the pools nearly constant in size).
- **The call is CARRY_PL6 on all seven** (BELOW 0, ABOVE 2, LEVEL 5). Under the user's later ruling (head of this section) it is the name of the reading, not a decision to take a cell above 2,500 downstream; PL7 reads the carry at 2,500 only.


**Which datasets meet the target on eval, frozen lists (ALL >= 0.95), of the 7 read: 6**: MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MuSiQue, MetaQA. **They do not:** WebQSP (0.9474).

**The price, filed beside the call and deciding nothing** (eval, frozen lists; nodes per question of the pool the rule hands to the next stage):

| dataset | NEW mean nodes | OLD mean nodes | extra nodes per question (ratio) | NEW 95th percentile | NEW maximum | NEW mean as a share of the graph |
|---|---|---|---|---|---|---|
| MultiHop-RAG | 50.0 | 50.0 | +0.0 (1.00x) | 50 | 50 | 8.21% (609) |
| SQuAD | 15.0 | 15.0 | +0.0 (1.00x) | 15 | 15 | 0.07% (20,233) |
| HotpotQA | 77.9 | 77.9 | +0.0 (1.00x) | 125 | 125 | n/a |
| 2Wiki | 917.0 | 917.0 | +0.0 (1.00x) | 1,192 | 1,200 | n/a |
| MuSiQue | 15,400.0 | 2,438.2 | +12,961.8 (6.32x) | 15,400 | 15,400 | 13.10% (117,534) |
| MetaQA | 4,972.8 | 2,335.9 | +2,637.0 (2.13x) | 5,016 | 5,019 | n/a |
| WebQSP | 39,529.5 | 2,094.6 | +37,435.0 (18.87x) | 40,259 | 40,342 | 1.52% (2,592,894) |

The share of the graph is filed where a record holds the graph's node count (MultiHop-RAG, SQuAD, MuSiQue, WebQSP; HotpotQA, 2Wiki and MetaQA hold none in these records). Two costs are different: the pool's size is what the next stage reads per question; the first stage's own cost is the walk, filed by the grid heads' seconds (section 13.1) and the two-question probes (15.0 s per question for all 156 cells of MuSiQue's rung 2, 14.1 s for WebQSP's rung 3), and it is not read from the pool's size.


### 13.4 What the ladder bought beside the call (U95 and U98 of all cells against rule U over rung 0's 6,078 cells; eval; decides nothing)

| dataset | floor | ladder + rung 0 (rule U, eval) | rung 0 alone (rule U, eval) | ALL difference, 95% CI | nodes ratio, 95% CI | status (ALL-only) |
|---|---|---|---|---|---|---|
| MultiHop-RAG | U95 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| MultiHop-RAG | U98 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| SQuAD | U95 | `rrf1000_b15` 0.9554 at 15.0 | `rrf1000_b15` 0.9554 at 15.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| SQuAD | U98 | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| HotpotQA | U95 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| HotpotQA | U98 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| 2Wiki | U95 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| 2Wiki | U98 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9785 at 6,275.0 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9785 at 6,275.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| MuSiQue | U95 | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` 0.9640 at 15,400.0 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 | +4.18 [+3.27, +5.13] | 2.6818 [2.6770, 2.6860] | ABOVE (ABOVE) |
| MuSiQue | U98 | `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu` 0.9710 at 20,235.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 | +4.88 [+3.97, +5.83] | 3.5239 [3.5186, 3.5292] | LEVEL (ABOVE) |
| MetaQA | U95 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9582 at 4,972.8 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9582 at 4,972.8 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| MetaQA | U98 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9615 at 6,259.3 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9615 at 6,259.3 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (LEVEL) same cell |
| WebQSP | U95 | `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` 0.9474 at 39,529.5 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9175 at 6,079.5 | +2.99 [+2.13, +3.93] | 6.5021 [6.4742, 6.5307] | LEVEL (ABOVE) |
| WebQSP | U98 | `rrf1000_b1600+FULL:h3_bf_v40000~cs-vr` 0.9481 at 40,164.1 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9175 at 6,079.5 | +3.06 [+2.13, +3.99] | 6.6064 [6.5769, 6.6339] | LEVEL (ABOVE) |

On **MuSiQue** the ladder buys +4.18 points [+3.27, +5.13] at 2.68 times the nodes of the best rung-0 cell (rung 0's rule-U pick is `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr`, 0.9222 at 5,742.4: the any-size reading of PL5), and U98 reaches 0.9710 at 20,235.7 (the uncut V = 20,000 walk). On **WebQSP** it buys +2.99 points [+2.13, +3.93] at 6.50 times the nodes (0.9474 at 39,529.5 against 0.9175 at 6,079.5): ALL-only ABOVE, LEVEL by the rule, and neither floor is reached on eval (U98 0.9481 at 40,164.1). On the five datasets that ran no rung (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MetaQA) the ladder buys nothing: the same cell on both sides, difference exactly 0.


### 13.5 The frontier above the bound (matched node counts; eval ALL difference of the ladder's cells against rung 0's, points; decides nothing)

A dataset is BELOW if any budget is BELOW, ABOVE if at least two budgets are ABOVE and none BELOW, else LEVEL (ALL-only status; `=` marks the same cell on both sides):

| dataset | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 | 10,000 | 15,000 | 20,000 | 30,000 | 40,000 | 60,000 | 80,000 | frontier status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| SQuAD | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| HotpotQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| 2Wiki | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| MuSiQue | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +1.28 [+0.58, +2.03] ABOVE | +2.07 [+1.57, +2.65] ABOVE | +4.39 [+3.52, +5.34] ABOVE | +4.88 [+4.01, +5.79] ABOVE | +4.88 [+3.97, +5.79] ABOVE | +4.88 [+3.85, +5.75] ABOVE | +4.88 [+3.97, +5.75] ABOVE | **ABOVE** |
| MetaQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| WebQSP | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +1.53 [+0.93, +2.20] ABOVE | +1.53 [+0.93, +2.26] ABOVE | +2.46 [+1.73, +3.19] ABOVE | +2.99 [+2.13, +3.93] ABOVE | +3.06 [+2.20, +3.99] ABOVE | +3.06 [+2.26, +3.93] ABOVE | **ABOVE** |

- **Frontier counts:** BELOW 0, ABOVE 2, LEVEL 5 (read 7 of 7).

Where the ladder's cell and rung 0's differ (dev ALL at dev nodes; eval ALL at eval nodes):

| dataset | budget | cells | dev | eval | difference |
|---|---|---|---|---|---|
| MuSiQue | 10,000 | ladder `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu+top7500`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9420 at 9,086.3; 0.9355 at 5,783.4 | 0.9350 at 9,076.2; 0.9222 at 5,742.4 | +1.28 ABOVE |
| MuSiQue | 15,000 | ladder `rrf1000_b1600+FULL:h2_c100_v20000~s20-cs-rk+top15000`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9492 at 13,970.0; 0.9355 at 5,783.4 | 0.9429 at 13,614.8; 0.9222 at 5,742.4 | +2.07 ABOVE |
| MuSiQue | 20,000 | ladder `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu+top15000`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9622 at 16,586.3; 0.9355 at 5,783.4 | 0.9661 at 16,576.2; 0.9222 at 5,742.4 | +4.39 ABOVE |
| MuSiQue | 30,000 | ladder `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9668 at 20,264.2; 0.9355 at 5,783.4 | 0.9710 at 20,235.7; 0.9222 at 5,742.4 | +4.88 ABOVE |
| MuSiQue | 40,000 | ladder `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9668 at 20,264.2; 0.9355 at 5,783.4 | 0.9710 at 20,235.7; 0.9222 at 5,742.4 | +4.88 ABOVE |
| MuSiQue | 60,000 | ladder `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9668 at 20,264.2; 0.9355 at 5,783.4 | 0.9710 at 20,235.7; 0.9222 at 5,742.4 | +4.88 ABOVE |
| MuSiQue | 80,000 | ladder `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu`; rung 0 `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9668 at 20,264.2; 0.9355 at 5,783.4 | 0.9710 at 20,235.7; 0.9222 at 5,742.4 | +4.88 ABOVE |
| WebQSP | 15,000 | ladder `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu`; rung 0 `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9416 at 10,810.7; 0.9257 at 6,070.5 | 0.9328 at 10,817.4; 0.9175 at 6,079.5 | +1.53 ABOVE |
| WebQSP | 20,000 | ladder `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu`; rung 0 `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9416 at 10,810.7; 0.9257 at 6,070.5 | 0.9328 at 10,817.4; 0.9175 at 6,079.5 | +1.53 ABOVE |
| WebQSP | 30,000 | ladder `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu`; rung 0 `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9496 at 20,744.3; 0.9257 at 6,070.5 | 0.9421 at 20,737.5; 0.9175 at 6,079.5 | +2.46 ABOVE |
| WebQSP | 40,000 | ladder `rrf1000_b400+FULL:h3_bf_v40000~cs-vr`; rung 0 `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9555 at 39,571.6; 0.9257 at 6,070.5 | 0.9474 at 39,529.5; 0.9175 at 6,079.5 | +2.99 ABOVE |
| WebQSP | 60,000 | ladder `rrf1000_b1600+FULL:h3_bf_v40000~cs-vr`; rung 0 `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9569 at 40,201.1; 0.9257 at 6,070.5 | 0.9481 at 40,164.1; 0.9175 at 6,079.5 | +3.06 ABOVE |
| WebQSP | 80,000 | ladder `rrf1000_b1600+FULL:h3_bf_v40000~cs-vr`; rung 0 `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9569 at 40,201.1; 0.9257 at 6,070.5 | 0.9481 at 40,164.1; 0.9175 at 6,079.5 | +3.06 ABOVE |

- **MuSiQue's frontier (eval) rises from 7,000 nodes upward**: the matched cells equal rung 0's up to 7,000, then 0.9350 at 9,076 (+1.28 points, ABOVE), 0.9429 at 13,615 (+2.07), 0.9661 at 16,576 (+4.39) and 0.9710 at 20,236 (+4.88). From 30,000 to 80,000 the cell is the same and the difference stays at +4.88: **that flat tail is where the ladder stopped (REACHED at rung 2), not a measurement that more nodes would buy nothing.**
- **WebQSP's** matched cells equal rung 0's up to 10,000 nodes, then 0.9328 at 10,817 (+1.53 points), 0.9421 at 20,738 (+2.46), 0.9474 at 39,530 (+2.99) and 0.9481 at 40,164 (+3.06): about +1.5, +0.9 and +0.5 points of eval ALL for each step of 1.8 to 1.9 times the nodes. The 60,000 and 80,000 cells are the V = 40,000 cell, again where the ladder stopped.
- **The shapes differ**: WebQSP's gain per step falls (+1.5, +0.9, +0.5 points); MuSiQue's is uneven, its largest single step (+2.32 points: 0.9429 at 13,615 to 0.9661 at 16,576) falling between 15,000 and 20,000 budgets and its last (+0.49, to 0.9710 at 20,236) at the ladder's end. The frontier reading counts MuSiQue and WebQSP ABOVE (two or more budgets ABOVE and none BELOW). MetaQA ran no rung, so its frontier is rung 0 read against itself (the same cell at every budget, difference exactly 0).


### 13.6 What is still short, in numbers (diagnostics D1 and D2, post-hoc, labels read, 300 dev questions evenly spaced over the carve; decides nothing)

**None to report.** Section 7 of the declaration asks for D1 and D2 at the chosen cell of a dataset that ends STOPPED_RISING or LADDER_END; every ladder here ended REACHED on dev, and `zpl5r.py miss --pl6` skips a dataset whose pick reaches 0.95 (it runs only with `--all`). PL5's D1 and D2 at the bounded picks (section 14 of PL5's page) stand as the account of what the 2,500 bound cuts. **The one shortfall left is WebQSP on eval**: dev 0.9555, eval 0.9474; a gap of 0.8 points, which is the kind selection over about 6,700 cells on a 1.5k-question carve produces (section 12 of the declaration); no diagnostic was run to separate it from a difference between the two carves, and none reads what the walk misses.

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes |
|---|---|---|---|---|---|---|---|


| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 1 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|---|



### 13.7 What the call means, and what it leaves unread

- **What the map says.** The three datasets that PL2 to PL5 left SHORT inside 2,500 nodes reach the target on dev above it: MetaQA at 4,984 nodes on dev (4,973 on eval, 2.13 times the bounded pick's), MuSiQue at 15,400 (6.32 times), WebQSP at 39,572 (39,530 on eval, 18.87 times). On eval MetaQA (0.9582) and MuSiQue (0.9640) meet 0.95; WebQSP (0.9474) does not. The cost of coverage on these three is not small and not flat: MuSiQue's frontier stands 4.88 points above rung 0's at 20,236 nodes and WebQSP's 3.06 points above at 40,164, and WebQSP's gain per step of nodes is falling.
- **What the bound was doing.** At 2,500 nodes the three datasets read 0.8846, 0.8968 and 0.8789 on eval; the extra points are bought at nodes per question that are 13.1% of MuSiQue's graph at the pick and 1.5% of WebQSP's. The first four datasets are untouched by lifting the bound (their picks are inside it and rule U names them).
- **What it leaves unread.** (a) No cell above 2,500 is read on e3v lists here (the beside-read `zpl6g.py confirm --first e3v` was not run; it decides nothing). (b) No rung beyond 2 (MuSiQue) or 3 (WebQSP) was run, so the frontier above 20,000 and 40,000 nodes is unread, and WebQSP's eval shortfall of 0.26 points is not mapped. (c) The pool size is the mean of the pool the walk hands on; the walk's cost is filed by the grid heads' seconds and the two-question probes only. (d) The links are the current ones: the UL2 per-label table (cited in section 1 of PL7's page) puts 98.7% of WebQSP's candidate pairs in one label at 0.009% precision, so a smaller pool at the same ALL is partly a question for the link set, which the next stages address (PL7, then the link-tightening stage and one pool re-run).


### 13.8 The next changes (names only; none is adopted)

**Declared before any number: PL7** (`docs/PL7_COVERAGE_RESTRICTION.md`, 24f4892e): restrict the pool back toward 2,500 nodes under one rule on all seven (a ladder of ceilings from this page's picks down to 2,500; hard-cut cells; the carry is read at 2,500 only; the node cost and a floor caveat beside every reading). It is provisional on the current links. After it: a link-tightening stage (per-label and per-class keep-or-drop against gold reach, one rule on all seven, selected on dev carves and graded on eval), declared by the coordinator, then one pool re-run on the final link set. Names only here: a parent-conditioned rank, a hop-weighted schedule and a window wider than 4,096 (PL5 section 14) stay unadopted.


### 13.9 Surprises and contradictions with filed numbers

- **WebQSP reaches 0.95 on dev and not on eval** (0.9555 against 0.9474) with 39,530 nodes per question: a dev ladder state of REACHED is not a reading of the target; the eval call is.
- **The flat tails of the frontier are the ladder's end, not the data's** (13.5): MuSiQue's cells at 30,000 to 80,000 are one cell, as are WebQSP's at 60,000 and 80,000.
- **The pools are nearly constant in size** (the 95th percentile and the maximum are within 2.1% of the mean at every new pick), because the pick is a hard cut or a budget-following walk that spends its budget; a mean node count is therefore close to every question's.
- **Nothing contradicts a filed number.** PL5's selects are rung 0 unchanged, and the laptop's PL5 confirmation grid and the host's PL6 confirmation grid agree bit for bit on the cells both hold (see 13.10).


### 13.10 Disclosed and undone
- **The two confirmations ran as host rows, not on the laptop** (the declaration's section 6 says the laptop reads them): `pl6-musiquecf` (2,417 eval questions) and `pl6-webqspcf` (1,503), so as not to wait for the laptop, whose chain was still running PL5's reads (one heavy laptop job at a time). The host builds only the cells PL5's confirmation grid does not hold. **Cells held by both agree exactly**: WebQSP 7 cells over 1,503 questions and MuSiQue 5 cells over 2,417, 0 differing (present counts and pool sizes, element for element; `grid_eval_frozen_confirm.*` against PL5's laptop grid).
- **G-R ran on the host for MuSiQue only** (`gr_musique_host.json`), as section 8 of the declaration says ("host: the host's G-R on MuSiQue, as PL5's"); the WebQSP rungs depend on that row in the feed and have no G-R of their own. The laptop ran none for PL6 (it ran no rung).
- **Rows of later rungs were added to the feed after the stop rule asked for them**: MuSiQue rung 2 and WebQSP rungs 2 and 3 were appended to `outputs/pl6/feed_rows.txt` after the previous rung's select (section 6: "rung 2 and later rows are not in the file"), and `feed_rows_confirm.txt` holds the two confirmation rows; no cell, rule or constant changed.
- **This page is an analysis**, by the user's ruling relayed before PL7 (head of this section); the call is filed under its rule's name and carries nothing above 2,500 downstream.
- The grid arrays (`*.npz`) are not committed (as PL2 to PL5); the heads, selects, confirmations, the call, the probes and this page are.

