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
