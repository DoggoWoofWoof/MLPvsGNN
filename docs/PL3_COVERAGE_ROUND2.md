# PL3: the coverage target, round 2 (the six candidate changes of docs/PL2_COVERAGE_TARGET.md section 10.6, on all seven)

Declared 25 November 2026, before any PL3 coverage number exists. What I have read are PL2's FILED records
(`outputs/pl2/<d>/select.json`, `confirm_frozen.json`, `confirm_e3v.json` for MultiHop-RAG, `outputs/pl2/call.json`, PL2's page
sections 10.1 to 10.7; commits aa3e486 declaration, 656cc37 engine, b251d67 results). Nothing below was computed after the page's
rule was chosen. The only runs before this commit are label-free or agreement checks: the cost of each walk (`zpl3g.py time`, which
prints seconds and nodes scored, no label is read), the walk gate G-X3 on MultiHop-RAG and SQuAD (the engine's walk arrays against a
per-node loop; no label), and the engine self-test (the engine against a brute-force pool builder; it prints agreement counts, not
coverage). **Nothing here trains, no model is fitted, no LLM or spaCy is used, and every pool function is gold-free.** The pools of
record, PL1's pools and PL2's code and records are untouched. This page is the second round of the coverage target
(docs/PL1_POOLS_ON_UL2_LAYOUT.md section 7): it names the arms it targets (all nine), the changes it tries (section 4) and it enters
by UL2's universal rule (section 6).

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how
  much ever is possible".
- Earlier, still binding: "the pools or the rules or links all should be universal"; "i dont want to freeze pools for the sake of
  it, make sure we freeze when we know it is the best for that specific arm"; "link improvement with precision is all we need".
  Pools first and parameter-free (no learning, no model, no LLM, no spaCy); one model per family; the hub line is CLOSED (UL2
  8.15): none of the changes below reads a node's degree or adds a hub step; every rule is universal on all seven datasets
  (MultiHop-RAG, MuSiQue, SQuAD, MetaQA, WebQSP, HotpotQA, 2Wiki), carried everywhere or nowhere. Standing order: do not wait,
  fix what comes up, keep the cpu busy.

## 1. What PL2 found (the records this round starts from)

PL2's rule (smallest mean nodes with dev ALL >= 0.95 inside the 2,500-node bound, else the highest ALL) read on its 804 cells,
frozen first stage, selected on dev and confirmed on eval (`outputs/pl2/<d>/select.json`, `confirm_frozen.json`):

| dataset | PL2's chosen cell (R95 over all 804 cells) | dev ALL at nodes | eval ALL (frozen) at nodes | state |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 at 50.0 | 0.9690 at 50.0 | past 0.95 |
| squad | `rrf1000_b15` | 0.9519 at 15.0 | 0.9554 at 15.0 | past 0.95 |
| musique | `rrf1000_b800+FULL:h1_c100` | 0.8012 at 1,082.6 | 0.8283 at 1,064.2 | **SHORT** (best on the grid at any size 0.8449 at 2,611.7) |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` | 0.8958 at 2,157.6 | 0.8952 at 2,153.5 | **SHORT** (0.9613 only at 5,003.7, past the bound) |
| webqsp | `dense_top200+FULL:h2_c100_v2000` | 0.7505 at 1,888.2 | 0.7638 at 1,890.8 | **SHORT** (best at any size 0.8016 at 5,100.2) |
| hotpotqa, 2wiki | not read: their dev grids are host rows (`pl2-g-*`) | | | |

`outputs/pl2/call.json`: INCOMPLETE (5 of 7), BELOW 0, ABOVE 4, LEVEL 1. The diagnosis of PL2 section 10.5 (the filed `reachability`
blocks, M3A's populations) is what this round is built on:

- **The ceilings are far above the cells.** An uncapped three-hop FULL walk from the seeds holds 1.000 / 0.998 / 0.996 of the
  missing golds on MuSiQue / MetaQA / WebQSP (every missing gold lies within three hops for 1,003 of 1,003, 35,989 of 36,160 and
  1,053 of 1,074 of the questions that miss one). The best cells hold 0.80 to 0.90 of the questions. So the caps (per-seed,
  per-frontier, visited) and the order the visited cap is filled in cut before the gold; the reach is not the limit.
- **MetaQA and WebQSP are answer-entity questions:** a question that misses a gold misses about 7.7 and 8.1 of them, so ALL is the
  product of about eight cuts. NER and KNN reach under 4% of those golds within three hops (WebQSP NER 0.025, KNN 0.032; MetaQA NER
  0.000, KNN 0.017): on those two the typed structure is the whole result and the other two families only add nodes.
- **MuSiQue** needs FULL at three hops (0.892 at two, 1.000 at three) and its best cell is the deep base with the shortest walk.
- PL2 section 10.6 named six changes; **none was tried** (they changed the walk or its seeds, which the PL2 grid did not vary).
  They are this round's cells.

## 2. What is fixed (PL2's page, by reference)

Everything of PL2 sections 2, 3 and 4.1 to 4.2 is unchanged and read by the same code (`zpl2g.py` imported, not edited): the
substrate (the served package for the six datasets, MultiHop-RAG's own B0 layout), the dev carves (`s1sel`; MultiHop-RAG's dev
split; WebQSP's `sorted(train ids)[1::2]`) and eval carves (`s1eval`; MultiHop-RAG's test), the frozen first stage for selection and
the frozen and e3v first stages for confirmation, the seeds (dense top 5 U SPLADE top 5, first occurrence kept) unless a cell says
otherwise, the measures (ALL = `all_gold_at_pool`, ANY, mean nodes = `candidates_mean`, Wilson 95% on ALL, paired bootstrap of
1,000 resamples with `numpy.random.default_rng(20261125)`), the floors 0.95 and 0.98 and the bound **2,500** mean nodes. The walk of
the "off" cells is `m3b_pools.expand_hops`, as in PL2. The three things that change are: the cell list (PL2's 804 cells plus the
cells of section 5), the incumbent of the call (PL2's own chosen cell, section 6), and that every walk may now use the question
vector (section 4).

## 3. What this round tests, and what it does not

| PL2 10.6 candidate | here | cells |
|---|---|---|
| 1. similarity-ordered caps | option `cs` | 168 (tag `cs`), and inside the combined walks |
| 2. regime per hop (STRUCT at hops 2 and 3, NER and KNN confined to hop 1) | regime `FS` = FULL at hop 1, STRUCT at hops 2 and 3 | 252 (tag `fs`) |
| 3. deeper base before the walk | bases `rrf1000_b400`, `b800`, `b1600` (and `b25`, `b100`, `dense_top200`) with PL2's walks, ranked sizes M up to 1,600 | 254 (tag `off`, 56 of them with the `b1600` base) |
| 4. seed widening | options `s10`, `s20`: seeds = the first 10 or 20 of the fused list | 252 (tag `seeds`) |
| 5. bridge-conditioned ranking | ranked cells `+brg{M}` | 40 (tag `brg`) |
| 6. visited cap by rank | option `vr` | 252 (tag `vr`), and 252 with `cs` (tag `csvr`) |

**Label-ordered caps** (the caps taken in the UL2P ledger's precision order, K.knn first) are not in this round: the package graph
carries no per-link rule label (PL2 section 2a: the UL2 layout stores are not materialized), so the order is not expressible
without a layout; it is **named as the next round**, with support-ordered expansion (the number of seeds that reach a node) and
reading the per-link weights, and it goes on its own page. No link-rule change is made here (it goes through UL2's gates).

## 4. The changes, exactly (the walk; section 8's gates hold the code to this text)

A **walk** is `{regime}:{setting}` or `{regime}:{setting}~{mods}`, mods joined by `-` in the order `s10`, `s20`, `cs`, `vr`
(`s10` and `s20` never together; `vr` never on a one-hop setting). The regimes are PL2's STRUCT, NER, KNN, FULL (all three
families) and the new **FS**. The settings are PL2's (hops / per_seed_cap / per_frontier_cap / visited_cap): `h1_c100` 1/100,
`h2_c100_v5000` 2/100/100/5000, `h3_c100_v5000` 3/100/100/5000, `h3_c25_v2000` 3/25/25/2000. A walk with no mods and a regime
other than FS is PL2's walk, bit for bit (gate G-R).

- **cs (similarity-ordered caps).** At every hop the cap keeps, for each node and family, the `cap` neighbours with the highest
  float64 inner product with the question vector, out of the node's first **512** neighbours in CSR row order (`SCORE_WINDOW`; a
  row longer than 512 is cut to its first 512 before ranking: a disclosed limit), ties to the lower node id, best first. The output
  order is `capped_multi`'s: node-major, then family, then rank. The rest of the walk (first occurrences, visited skipped, the
  visited cap) is unchanged.
- **vr (visited cap by rank).** At hops 2 and 3 the fresh nodes of a hop (not yet visited, first occurrence) are ordered by inner
  product with the question vector (ties to the lower id) **before** the visited-cap budget is applied, so the cap drops the
  lowest-cosine fresh nodes instead of the last to arrive; the surviving nodes are the next frontier, expanded in that order.
- **FS.** Hop 1 walks all three families, hops 2 and 3 walk STRUCT only.
- **s10, s20.** The seeds are the first 10 (20) nodes of the fused list `F1` = equal-weight RRF (constant 60) of the dense and
  SPLADE top-1000 lists, in place of dense top 5 U SPLADE top 5.
- **+top{M}** (PL2's rule): the pool is the base list's prefix U the top M of the walk's nodes that are not in the base, ranked by
  float64 inner product with the question vector, ties to the lower node id. **+brg{M}** (candidate 5): the same with the
  ranking vector `r = q + t / |t|`, where `t` is the mean of the vectors of the fused list's first 3 documents (the same encoder
  as `q`); the walk itself is unchanged, only the order of its nodes differs.
- **Inner products** are accumulated in float64 from the fp16 rows in numpy's pairwise order (no BLAS), so the order does not
  depend on the thread count (gate G-D); one score per node and question is computed once and remembered for that question.
  The vectors are the first stage's own encoder (the package's frozen dense vectors for the selection read; e3v's document and
  query vectors for the e3v confirmation), as in PL2 section 2b. No second encoder, no encode, no GPU.
- **Unions** (no `+top`): the base U the whole walk, as PL2's G cells.

## 5. The cells (fixed at this commit; nothing is added after a number is read)

- Bases: `rrf1000_b25`, `rrf1000_b100`, `rrf1000_b400`, `rrf1000_b800`, `rrf1000_b1600`, `dense_top200` (RRF of the dense and SPLADE
  top-1000 lists, a prefix of b rows; the dense list's top 200). Ranked sizes M: 50, 100, 200, 400, 800, 1,600.
- **36 walks** (`zpl3g.py cells` prints them): STRUCT and FULL on `h2_c100_v5000` with mods {-, cs, vr, cs-vr}; STRUCT and FULL on
  `h3_c100_v5000` and on `h3_c25_v2000` with mods {-, vr, cs-vr}; FS on `h2_c100_v5000`, `h3_c100_v5000` and `h3_c25_v2000` with mods
  {-, cs-vr}; STRUCT and FULL on `h1_c100` with mods {-, cs}; the seed walks FULL:`h1_c100`~s10 and ~s20, FULL:`h2_c100_v5000`~s10-cs-vr
  and ~s20-cs-vr, STRUCT:`h3_c25_v2000`~s10-cs-vr and ~s20-cs-vr. (The walk list was cut from 40 to 36 after a cost timing of
  one dataset, before any coverage was read: `cs` alone at three hops was dropped, `cs` alone is read at two hops, and the STRUCT
  seed walks moved to the 25-cap setting.)
- Each walk x each base as a union (6) and as a ranked cell (6 bases x 6 sizes = 36) = 42 cells; the **bridge-ranked cells** are 10
  walks (STRUCT and FULL on `h1_c100`, and on `h2_c100_v5000` and `h3_c100_v5000` each with mods {-, cs-vr}) x bases `rrf1000_b100`,
  `b400` x M in {100, 400} = 40. Of 36 x 42 + 40 = 1,552 names, **82 are PL2's own cells and are not repeated: 1,470 new cells**.
- The new cells' sha256 (`P2.cells_sha256`, the sorted names): `ff879881090993bca1e063b715724c0d89411b8cacc77f5df7bcccc2b29c642e`
  (1,470 distinct). PL2's 804: `76bdf3a822e6bb8caf21121f853c9a8383e7844f99425e004d18c75f2aededc1`. The union, 2,274 distinct:
  `62767f7326bbbda0ad846ab0492466b8abd51f2f634d1ff6746afeb7c4acb093`.
- Tags (descriptive, for what each family bought): `cs` 168, `vr` 252, `csvr` 252, `fs` 252, `seeds` 252, `off` 254 (the base-depth
  cells), `brg` 40; `deep` = the `rrf1000_b1600` base (a second tag on 252 cells). The tags sum to the 1,470.
- Cells above the bound are read, cannot be selected, and are shown in the any-size readings.

## 6. The rule, the confirmation and the call (PL2 section 4, the same, with PL2's pick as the incumbent)

**Selection** (dev, frozen lists, over the union of PL2's 804 and the 1,470): among cells with mean nodes <= 2,500, **R95** = the
smallest mean nodes with dev ALL >= 0.95 (ties to the higher ALL, then the name); if none, the highest ALL (ties to fewer nodes),
and the dataset is flagged SHORT. **R98** likewise at 0.98. Readings (each by the same rule): `all` (the decider), `pl2` (PL2's
804 only, which must give PL2's filed pick: checked against `outputs/pl2/<d>/select.json`), `new` (the 1,470 only), and per tag
`pl2+TAG` (PL2's cells plus one family) and `only:TAG`. The whole curve and the frontier of each reading are filed.

**Confirmation** (eval, nothing re-chosen): the cells R95 and R98 name in the readings `all`, `pl2` and `new` (at most six) and
PL2's filed own construction are re-run unchanged on the eval carve, with the frozen lists and with e3v's lists. CONFIRMED = eval
ALL >= the floor; otherwise SHORT by x points.

**The call.** On eval, frozen lists, per dataset: NEW = R95 of the reading `all`, OLD = R95 of the reading `pl2` (PL2's own chosen
cell, the incumbent). Paired bootstrap of the ALL difference (NEW - OLD) and of the ratio of mean nodes (NEW / OLD).
- **BELOW**: NEW misses 0.95 where OLD reaches it, or the ALL difference's upper end is below 0 with the nodes ratio's lower end at
  or above 1.
- **ABOVE**: NEW reaches 0.95 where OLD does not, or the nodes ratio's upper end is below 1 with ALL not BELOW, or the ALL
  difference's lower end is above 0 with the nodes ratio's upper end at or below 1.
- **LEVEL**: anything else (a dataset whose R95 is PL2's cell is LEVEL by construction).

**CARRY_PL3** (the union grid replaces PL2's 804 cells as the own-construction rule of all seven datasets; each dataset's R95 cell
is its construction) when all seven are read, no dataset is BELOW, at least one is ABOVE, and every chosen cell is within the 2,500
bound. **INCOMPLETE** while fewer than seven are read. Otherwise **NOT_CARRIED: PL2's grid stays the rule** and PL3's cells are
filed as the diagnosis. One universal rule, carried everywhere or nowhere: the per-tag readings and the R98 and `new`-only calls are
filed beside the call and decide nothing.

## 7. Stop rule

The 36 walks, the bases, the sizes M, the settings, the bound, the floors, the readings and the dev and eval carves are fixed by this
commit; every output records the sha256 of the cell list and the code-path hash, and `zpl3g.py` refuses a grid whose cell list is not
the declared one and a select whose heads carry another code-path hash. Nothing is added to the grid after a number is read. If a dataset is still SHORT after this round the report
(a) says in numbers which gold nodes remain unreached (section 10), and (b) names the next change, which is declared on its own
page (section 12 lists the names).

## 8. Gates (each is a failure that stops that dataset's reading; none of its numbers is read)

- **G-R (PL2's cells are reproduced).** The PL3 engine, given PL2's 804 cell names, builds them on the first 400 dev questions of a
  dataset and every present-count and pool-size equals PL2's filed dev array element for element. Required on MuSiQue and MetaQA
  at least (run on all five laptop datasets); the host's own G-R on MuSiQue is a precondition of every host select and
  confirm (`pl3-gr-musique`).
- **G-X3 (the walk).** With every option off the engine's walk equals `m3b_pools.expand_hops` array for array, on 60 questions, for
  every regime and setting; with every option, each of the 36 walks equals an independent per-node Python loop written from
  section 4, on 3 questions per dataset.
- **Engine self-test.** The engine's present counts and pool sizes against a brute-force pool builder written from section 4
  (explicit sets) on a sample of questions, 3 cells per walk.
- **G-F (gold-free).** No pool function takes or reads a label (checked on the signatures and the source of every function that
  builds a pool); the word does not occur in their sources; the states are built, and hashed for G-D, from a source whose label arrays are
  withheld. Labels enter only in `grade3`.
- **G-D (determinism).** The pools of a sample of questions hash the same under 1 thread with chunk 64 and 4 threads with chunk 7.
- **G-U (universal).** One code-path sha256 (the pool code, the constants and PL2's code-path hash) and the cell list's sha256 are
  recorded in every grid head; `select` refuses a head that differs and `call` refuses to file when two datasets differ.

## 9. What each family can and cannot show

| family | what it can show | what it cannot show |
|---|---|---|
| `cs` similarity caps | whether the cap that now drops golds in CSR order drops fewer when it keeps the neighbours closest to the question; the effect is largest where a row is longer than the cap (cap 25 and cap 100 on hubs) | a gold outside the first 512 neighbours of a long row (the window); a gold that is far from the question in the encoder's geometry (the bridge document, whose own text does not look like the question); it uses no label information |
| `vr` visited cap by rank | whether, once the 5,000 (2,000) visited nodes are spent, the best fresh nodes survive rather than the first to arrive; only binding when the walk hits the cap, so it is silent where the cap does not bind | anything below the cap; the next frontier is better ordered but is the same set when the cap does not bind |
| `fs` regime per hop | whether dropping NER and KNN after hop 1 frees the cap for STRUCT at hops 2 and 3 (MetaQA, WebQSP: NER and KNN reach under 4%) | a gold that only NER or KNN reaches at hops 2 and 3 (MuSiQue: NER reaches 0.627 and KNN 0.256 within 3 hops); one split only, not every split |
| deeper base (`off` with b400 to b1600) | whether the first stage's deep tail plus PL2's walks holds more golds per node; the retrieval-only frontier is already filed, so this is the same walk on a deeper base | a gold that both first-stage lists rank past 1,000; a base above the bound |
| `s10`, `s20` seeds | whether more first golds to walk from help (MuSiQue ANY 0.999, ALL 0.80) | a gold not within the walk of any of the 10 or 20 seeds; more seeds spend the same caps, so MetaQA and WebQSP may lose |
| `brg` bridge ranking | whether the second gold, which the question text alone does not point at, ranks higher when the best first-stage documents are added to the query | a gold the walk does not reach; the added vector is the fused list's own top 3, so a wrong top 3 drags the ranking; two bases and two sizes only |

None of the six can reach a gold the walk does not reach (the ceilings of section 1) or a gold that is unreachable in the first stage
and the graph together; the pools and links are rule-made, no label is read, and no semantic signal is added that the first stage's
encoder did not have.

## 10. The short datasets: what remains unreached (diagnostic D1; post-hoc and labelled so; it decides nothing)

For each dataset whose R95 (reading `all`) is SHORT, at that chosen cell, on dev: each gold node missing from the pool of a question
gets one class by this order, and the report gives the number of missing golds in each class and the share of questions that miss at
least one gold of each class:
- **U (unreached):** not in the uncapped three-hop FULL reach of the seeds (no per-node cap, no visited cap) and not in the cell's
  base list: no cap or ranking could have held it; it is the reach ceiling.
- **C (cut by the caps):** in that uncapped reach but not in the cell's own walk (the per-seed, per-frontier or visited cap, or the
  order, dropped it).
- **R (cut by the ranking or the bound):** in the cell's own walk but not in the pool (its rank is beyond M).
The report puts these beside the pool's mean nodes and the 2,500 bound, and says which of the three the next change has to attack.
The code is `outputs/mp_unified/zpl3r.py miss`; it reads the labels, so it is run after `select` and its numbers are never an input
to any rule.

## 11. Code, rows, outputs

- **New modules only:** `outputs/mp_unified/zpl3g.py` (engine, commands `cells`, `time`, `gr`, `grid`, `select`, `confirm`, `call`,
  `selftest`; every command except `cells` and `call` takes `--host`) and `outputs/mp_unified/zpl3r.py` (the report and D1). They
  import `zpl2g.py` (the sources, lists, grading helpers, selection, paired bootstrap), `scripts/m3a_headroom.py` and
  `mp_retrieval.m3b_pools` and edit none of them.
- **Outputs:** `outputs/pl3/<d>/{grid_dev_frozen*.npz (arrays, uncommitted), grid_dev_frozen*.json (heads), select.json,
  confirm_frozen.json, confirm_e3v.json}`, `outputs/pl3/{cells.json, selftest.json, gr_<d>.json, call.json, report.md}`.
- **Laptop:** MultiHop-RAG, SQuAD, MetaQA, MuSiQue, WebQSP, one heavy job at a time (three parts of the dev grid in parallel),
  frozen lists, and e3v for MultiHop-RAG (the other datasets' e3v vectors sit on the host). **Host rows** (`pl3-*`, CPU only, 2 cpus
  for the small rows, in `outputs/pl3/feed_rows.txt`, written after the module is committed): the host's G-R on MuSiQue; HotpotQA
  and 2Wiki dev grids (three parts each), select and confirm (they depend on PL2's `pl2-g-*` rows, whose dev grids the union
  reads); the e3v confirmations of SQuAD, MuSiQue, MetaQA and WebQSP.

## 12. What follows

1. The call (section 6) over all seven; a result section is filed here (section 14) with, per dataset, PL2's chosen cell against the
   new frontier, the smallest cell at 0.95 and at 0.98 (dev and eval), what each family bought, what is still short and D1.
2. **Next changes, names only** (none is adopted, none is in the grid): label-ordered caps (on a UL2 layout, per-link rule labels
   ordered by the ledger's precision), support-ordered expansion (the number of seeds that reach a node), reading the per-link
   weights, a hop-3 frontier restricted to the nodes the hop-2 ranking kept, per-family visited budgets.
3. The same grids on the UL2 layout stores once S1 has materialized them, and on e3v for dev once e3v lists exist for the select
   carves.

## 13. Disclosed

- Written after reading PL2's filed records and its page (sections 1 to 10), the walk code of `m3b_pools`, and the cost timings
  and gates named at the top; no PL3 coverage number was computed before this commit.
- 2,274 cells are selected over on 1.5k-question dev carves (224 for MultiHop-RAG, about +/- 2 points): the winner's dev ALL is
  optimistic, more so than PL2's over 804 cells. That is why the eval read decides and why the call is a paired interval on eval,
  against PL2's own chosen cell, not a comparison of dev numbers.
- PL3's cell list contains only c100 settings and `h3_c25_v2000`: the c25 hop-1 and hop-2 settings are PL2's. A family can look
  absent because its best setting is not in the grid; each family's tag reading says what it was given.
- `cs` ranks inside the first 512 neighbours of a row, `FS` is one regime split, `brg` is two bases and two sizes, and bases
  `b1200` (named in PL2 10.6) is not a base here: `b1600` stands for "deeper".
- The six datasets' graph is the old package graph (as PL2 disclosed); MultiHop-RAG reads its B0 layout. The UL2 layout is not
  materialized.
- The filed own constructions are not in the call: the incumbent is PL2's chosen cell, itself ABOVE the filed constructions on
  four datasets (PL2 section 10.4).

## 14. Results (25 Nov 2026; all seven datasets read, re-filed 08:40; this replaces the five-of-seven filing of commit 8bf1acf)

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl3/<d>/{select,confirm_frozen,confirm_e3v}.json`,
`outputs/pl3/call.json`) by `outputs/mp_unified/zpl3r.py report`; the tables of section 14.9 are pasted unchanged from
`outputs/pl3/report.md`. The five laptop datasets were read at 8bf1acf; HotpotQA and 2Wiki ran on the host (their dev grids, selects
and both confirmations) and the e3v confirmations of SQuAD, MuSiQue, MetaQA and WebQSP ran there too; the records came back into
`outputs/pl3/`. Nothing was changed in the rule, the cells or the engine after a number was read.

### 14.1 Gates and cost

The engine `zpl3g.py` is byte-identical to the declaration commit 7a55929: its sha256 starts `20ad63a08f5d` in all seven select
heads, the cell list `ff8798810909`, the code-path hash `e9d860950c70` is equal on all seven datasets (G-U PASS; `call.json`
`code_path_equal_everywhere` true). G-R (the engine rebuilds PL2's 804 cells element for element, 400 dev questions, 224 on
MultiHop-RAG) PASSED with 0 mismatches on the five laptop datasets and, as the precondition of the host reads, on the host for MuSiQue,
HotpotQA and 2Wiki (`gr_musique_host.json`, `gr_hotpotqa_host.json`, `gr_2wiki_host.json`, 321,600 elements each). The self-test
passed (G-X3, brute-force pools, G-F, G-D). The select of every dataset re-read PL2's own pick from its 804 cells and got PL2's filed
pick (`pl2_pick_equals_filed` true on all seven). Cost of the dev grids (sum of the three parts): 46 s MultiHop-RAG, 2,032 s SQuAD,
4,054 s MuSiQue, 1,982 s MetaQA, 4,122 s WebQSP (5.6 GB peak), and on the host 3,228 s HotpotQA (14.8 GB) and 4,819 s 2Wiki (17.1 GB).

### 14.2 The call: CARRY_PL3 (read 7 of 7: BELOW 0, ABOVE 2 (HotpotQA, 2Wiki), LEVEL 5), read clause by clause

On eval, frozen lists, per dataset: NEW = R95 of the reading `all`, OLD = PL2's own chosen cell (the incumbent), paired bootstrap of the
ALL difference and of the ratio of mean nodes (section 6). The clauses, each as the rule states it, for every dataset:

| dataset | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | BELOW: (1) NEW misses where OLD reaches / (2) ALL diff upper end < 0 with ratio lower end >= 1 | ABOVE: (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (ALL not BELOW) / (3) ALL diff lower end > 0 with ratio upper end <= 1 | status | chosen cell inside 2,500 (eval nodes) |
|---|---|---|---|---|---|---|---|
| MultiHop-RAG | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** | yes (50.0) |
| SQuAD | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** | yes (15.0) |
| HotpotQA | yes / yes | +0.07 [-0.07, +0.19] | 0.9996 [0.9994, 0.9997] | no / no | no / yes / no | **ABOVE** | yes (77.9) |
| 2Wiki | yes / yes | +1.06 [+0.74, +1.41] | 0.4285 [0.4272, 0.4295] | no / no | no / yes / yes | **ABOVE** | yes (917.0) |
| MuSiQue | no / no | +5.05 [+4.18, +5.92] | 2.3070 [2.2948, 2.3195] | no / no | no / no / no | **LEVEL** | yes (2,455.1) |
| MetaQA | no / no | +0.15 [+0.07, +0.24] | 1.0847 [1.0846, 1.0848] | no / no | no / no / no | **LEVEL** | yes (2,335.9) |
| WebQSP | no / no | +1.66 [-0.07, +3.39] | 1.3054 [1.2757, 1.3329] | no / no | no / no / no | **LEVEL** | yes (2,468.1) |

- **BELOW: none.** On no dataset does NEW miss the floor where OLD reaches it (clause 1), and no dataset has an ALL difference whose
  upper end is below 0 (clause 2): the smallest upper ends are the two identical pairs' 0.00 and HotpotQA's +0.19 points.
- **ABOVE: two.** **2Wiki** is ABOVE by two clauses: ALL +1.06 points [+0.74, +1.41] at 0.428 of the nodes [0.427, 0.430] (917.0 against
  2,140.1 on eval). The floor clause (1) does not fire on eval, because PL2's cell reaches 0.9503 there; on dev it would have (PL2's
  cell reads 0.9492 on dev, SHORT, and the new cell 0.9545 at 908.2). **HotpotQA** is ABOVE by clause 2 alone and by a thin margin, which
  is stated here as it is: the new cell is PL2's cell with the similarity-ordered cap `~cs`, 77.889 nodes against 77.921 on eval
  (ratio 0.99958, 95% CI [0.99942, 0.99972], an upper end below 1 by 0.03 percent), ALL +0.07 points with an interval that includes 0
  [-0.07, +0.19]. Clause 3 does not fire (the ALL interval's lower end is below 0). The rule is applied as written: ABOVE.
- **LEVEL: five.** MultiHop-RAG and SQuAD are PL2's own cells (LEVEL by construction); MuSiQue, MetaQA and WebQSP choose a different
  cell that buys ALL with more nodes (+5.05 points at 2.31 times the nodes, +0.15 at 1.085, +1.66 at 1.31): none of the ABOVE
  clauses fires, because each needs a ratio at or below 1 for a price read, and neither side reaches 0.95 for a floor read. That is
  the rule's design (section 14.7 states what it leaves unread), not a result of it.
- **The call.** CARRY_PL3 needs all seven read (yes, 7 of 7), no BELOW (0), at least one ABOVE (2) and every chosen cell inside the
  2,500 bound (yes: the largest is WebQSP's, 2,468.1 on eval and 2,442.4 on dev; MuSiQue's 2,455.1 and 2,488.9 on dev). **CARRY_PL3.**
  Section 5's union grid (PL2's 804 cells and the 1,470) replaces PL2's 804 cells as the own-construction rule of all seven datasets;
  each dataset's R95 cell of the reading `all` is its construction.
- **A reading beside the call (does not change it).** The call would be CARRY_PL3 on 2Wiki alone: if HotpotQA is set aside as LEVEL
  (its ABOVE rests on 0.04 percent fewer nodes), 2Wiki is ABOVE by two clauses, no dataset is BELOW, and every chosen cell is inside
  the bound.

### 14.3 What each dataset says (dev = selection carve, frozen lists; eval = confirmation, frozen and e3v lists)

The construction each dataset gets under CARRY_PL3 (R95 of the reading `all`), and PL2's, with eval nodes:

| dataset | PL3 construction (eval frozen ALL at nodes) | PL2's (eval frozen ALL at nodes) | meets 0.95 on eval, frozen / e3v |
|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | the same | yes / yes (0.9699) |
| SQuAD | `rrf1000_b15` 0.9554 at 15.0 | the same | yes / **no** (0.9421) |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | `rrf1000_b25+STRUCT:h1_c100+top100` 0.9627 at 77.9 | yes / yes (0.9642) |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | `equal_rrf_budget_200+FULL:h2_c25` 0.9503 at 2,140.1 | yes / yes (0.9684 at 916.1) |
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8788 at 2,455.1 | `rrf1000_b800+FULL:h1_c100` 0.8283 at 1,064.2 | no (short by 7.12 points) / no (0.8775) |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8952 at 2,153.5 | no (short by 5.32) / no (0.9296) |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7804 at 2,468.1 | `dense_top200+FULL:h2_c100_v2000` 0.7638 at 1,890.8 | no (short by 16.96) / no (0.7618 at 2,720.1) |

**Which datasets meet the target (eval ALL >= 0.95, frozen lists): four of seven**: MultiHop-RAG (0.969 at 50 nodes), SQuAD (0.955 at 15;
0.942 on e3v), HotpotQA (0.963 at 78) and 2Wiki (0.961 at 917; 0.968 on e3v). **On e3v three of seven** (SQuAD's 15-node pool falls short on
e3v, as PL2 filed). **They do not:** MuSiQue, MetaQA and WebQSP. In detail:

- **MultiHop-RAG** (224 dev questions; e3v read). R95 is PL2's cell (dev 0.9777 at 50.0; eval 0.9690 frozen, 0.9699 e3v). R98: PL2's
  `rrf1000_b60` (dev 0.9866 at 60.0; eval 0.9681 / 0.9664 e3v, short of 0.98 by 1.2 / 1.4 points) against PL3's
  `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (dev 0.9955 at 58.6; eval 0.9788 / 0.9779 e3v at 58.0 / 58.2 nodes, short by 0.12 / 0.21
  points): the R98 status is ABOVE, beside the call (R98 decides nothing). Highest ALL: 1.0000 at 100.0 nodes (PL2's
  `rrf1000_b50+NER:h3_c100_v5000+top50`), the same in both grids.
- **SQuAD** (1,498 dev questions). R95 and R98 are both PL2's cells (`rrf1000_b15`: dev 0.9519 at 15.0, eval 0.9554; and
  `rrf1000_b50+KNN:h1_c100+top25`: dev 0.9806 at 63.3, eval 0.9818 at 63.0; on e3v 0.9421 and 0.9763, both short): nothing new is
  smaller at either floor. The new cells buy only at large size: highest ALL inside the bound 1.0000 at 2,190.4 nodes
  (`rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800`) against PL2's 0.9980 at 1,544.8.
- **HotpotQA** (1,508 dev questions). PL3 `rrf1000_b25+STRUCT:h1_c100~cs+top100`: dev 0.9622 at 77.3 (PL2's cell without `~cs`: 0.9615 at
  77.3), eval 0.9634 / 0.9627 frozen at 77.9, e3v 0.9642 / 0.9646 at 77.5 / 77.6. The same pool at the same price: the similarity-ordered
  cap moves ALL by +0.07 points on dev and eval and by -0.04 on e3v (interval [-0.19, +0.08]). R98: PL3 `rrf1000_b100+STRUCT:h1_c100~cs+top200`
  (dev 0.9814 at 152.7; eval 0.9783 at 153.5, short of 0.98 by 0.17 points; e3v 0.9753) against PL2's 0.9776 at 154.5 (e3v 0.9743).
  Highest ALL inside the bound: PL2 0.9927 at 935.0, PL3 0.9960 at 2,246.8 (`rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr`). Dev frontier at equal
  nodes: +0.13 points at 150 nodes, +0.53 at 500, +0.66 at 750, +0.33 at 2,500, 0 elsewhere. The first-stage union already misses a gold
  on 378 of 1,852 questions and one STRUCT hop at cap 100 closes most of it (0.963 at 78 nodes); there is little left to buy.
- **2Wiki** (1,496 dev questions). PL2's pick is SHORT on dev (`equal_rrf_budget_200+FULL:h2_c25`: 0.9492 at 2,129.4; it reaches 0.9503 on
  eval frozen and 0.9750 on e3v at 2,140 / 2,139 nodes). PL3's `rrf1000_b400+FULL:h1_c100~s20+top800` reaches 0.95 on dev (0.9545 at
  908.2), 0.9609 on eval frozen at 917.0 and 0.9684 on e3v at 916.1: **the floor at 0.43 of PL2's nodes**. Smallest cell at 0.95 at any
  size on dev: PL2 `rrf1000_b800+FULL:h2_c25` 0.9545 at 2,695.6 (beyond the bound); PL3 908.2. At 0.98: none in either grid (best any
  size: PL2 0.9552 at 4,073.5, PL3 0.9719 at 6,230.3); inside the bound PL3's `rrf1000_b1600+FULL:h1_c100~s20+top800` 0.9606 on dev
  (eval 0.9656, e3v 0.9716). Dev frontier at equal nodes: +1.27 points at 750 nodes, +1.94 at 1,000, +1.74 at 1,500 and 2,000, +1.14 at
  2,500, 0 below 500.
- **MuSiQue** (1,534 dev questions; SHORT). PL2: `rrf1000_b800+FULL:h1_c100`, dev 0.8012 at 1,082.6, eval 0.8283 at 1,064.2 (e3v 0.8010),
  short by 12.17 points. PL3: `rrf1000_b800+FULL:h3_c25_v2000~cs-vr`, dev 0.8735 at 2,488.9, eval 0.8788 at 2,455.1 (e3v 0.8775 at
  2,457.0), short by 7.12 points. Smallest cell at 0.95 at any size: none; the highest ALL at any size is 0.9355 at 5,783.4 nodes
  (`rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr`), 2.3 times the bound and still below 0.95. Dev frontier at equal nodes (a reading of the
  filed tables, dev only): +3.06 points at 500 nodes, +5.22 at 750, +2.35 at 1,500, +4.37 at 2,000, +7.24 at 2,500.
- **MetaQA** (1,497 dev questions; SHORT). PL2: `equal_rrf_budget_200+STRUCT:h3_c25_v2000`, dev 0.8958 at 2,157.6, eval 0.8952 at 2,153.5
  (e3v 0.9281), short by 5.48 points. PL3: `rrf1000_b400+STRUCT:h3_c25_v2000`, dev 0.8978 at 2,339.7, eval 0.8968 at 2,335.9 (e3v 0.9296),
  short by 5.32 points. Smallest cell at 0.95 at any size: PL2 `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 nodes; PL3
  `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6, about twice the bound; at 0.98: none (best 0.9653 at 6,268.2). The frontier's +15.30
  points at 2,000 nodes is a lattice artifact, not a mechanism: PL2's frontier has no cell between 1,936.7 and 2,157.6 nodes, and PL3's
  `rrf1000_b25+STRUCT:h3_c25_v2000` (PL2's walk on a smaller base) reads 0.8938 at 1,999.3 against PL2's 0.8958 at 2,157.6.
- **WebQSP** (1,507 dev questions; SHORT). PL2: `dense_top200+FULL:h2_c100_v2000`, dev 0.7505 at 1,888.2, eval 0.7638 at 1,890.8 (e3v 0.7432),
  short by 18.62 points. PL3: `dense_top200+FULL:h2_c100_v5000~s10-cs-vr`, dev 0.7731 at 2,442.4, eval 0.7804 at 2,468.1, short by 16.96
  points; on e3v 0.7618 at 2,720.1 nodes (above the 2,500 bound on the e3v lists; the bound is read on the selection carve's frozen lists).
  Smallest cell at 0.95 at any size: none; highest ALL at any size 0.8480 at 6,067.3 nodes (`rrf1000_b1600+FULL:h3_c100_v5000~vr`). Dev
  frontier at equal nodes: +3.52 points at 300 nodes, +5.51 at 400 and 500, +6.04 at 750, +4.71 at 1,000, +6.37 at 1,500, +0.66 at 2,000,
  +2.26 at 2,500.

Selection optimism, measured: the picks moved from dev to eval frozen by +0.64 points (2Wiki), +0.12 (HotpotQA), +0.53 (MuSiQue), +0.73
(WebQSP), -0.10 (MetaQA), +0.35 (SQuAD) and -0.87 (MultiHop-RAG, 224 dev questions).

### 14.4 What each family bought (R95 of PL2's cells plus that family, dev; the best ALL inside the bound of that family alone)

- **Seed widening (`s10`, `s20`)** is what moved 2Wiki: PL2's R95 on dev is SHORT at 2,129.4 nodes, and PL2's cells plus the seed family give
  0.9545 at 908.2 (0.43 of the price); seeds also give WebQSP its largest step (0.7505 to 0.7731 at 2,442.4) and MuSiQue 0.8566 at 2,144.6.
- **Similarity-ordered caps (`cs`)**: HotpotQA's R95 (0.9615 to 0.9622 at 77.3, the call's thin ABOVE), MuSiQue 0.8611 at 2,394.6, WebQSP 0.7624
  at 2,289.1; **no change on 2Wiki** (PL2's cells plus `cs` = PL2; alone best 0.9432 at 1,799.3), MetaQA, MultiHop-RAG or SQuAD.
- **Both caps together (`cs` and `vr`)**: MuSiQue's largest step, 0.8012 to 0.8735 at 2,488.9; 2Wiki 0.9525 at 2,311.9; WebQSP 0.7624 at 2,288.5.
  **Visited cap by rank (`vr`) alone**: MuSiQue 0.8527 at 2,400.0, WebQSP 0.7545, 2Wiki 0.9559 at 2,334.0 (it reaches 0.95 on dev, at 2.6 times the
  price of the seed cell).
- **Deeper base before the walk**: MuSiQue 0.8605 at 2,386.3, MetaQA 0.8978 at 2,339.7 (its only mover, +0.20 points), WebQSP 0.7532, 2Wiki 0.9532 at
  2,350.9, HotpotQA none.
- **FS regime**: 2Wiki 0.9539 at 2,000.0, MuSiQue 0.8546 at 2,419.9, WebQSP 0.7724 at 2,367.7, MultiHop-RAG's R98 cell; none elsewhere.
- **Bridge-conditioned ranking bought nothing on any of the seven** (it only re-ranks the nodes the walk reached; D1's R class below is 0).
- MultiHop-RAG and SQuAD: no family moves R95 (the floor is reached at PL2's smallest cell).
- No family brings MuSiQue, MetaQA or WebQSP to 0.95 inside the bound. The best readings of R95 move dev ALL by +0.20 (MetaQA), +2.26
  (WebQSP) and +7.23 (MuSiQue) points, at 2,339.7, 2,442.4 and 2,488.9 mean nodes.

### 14.5 What is still short, in numbers (diagnostic D1, post-hoc, labels read; 300 dev questions evenly spaced over the carve)

Only MuSiQue, MetaQA and WebQSP are SHORT at R95 (HotpotQA and 2Wiki reach 0.95 inside the bound, so section 10 does not apply to them).
Each gold node missing from the pool of the chosen R95 cell has one class: **R** ranked past M, **C** not in the cell's own walk but in the
uncapped three-hop FULL reach of its seeds (the caps or the order dropped it), **U** in neither (the reach ceiling).

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (bound 2,500) |
|---|---|---|---|---|---|---|---|
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 36 of 300 (0.120) | 38 (1.06 per such question) | 0 | 38 | 0 | 2,487.2 |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | 33 of 300 (0.110) | 375 (11.36 per such question) | 0 | 373 | 2 | 2,337.4 |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 63 of 300 (0.210) | 245 (3.89 per such question) | 0 | 245 | 0 | 2,421.1 |

Of the 658 missing golds, 656 (99.7%) are C and 2 are U; none is R (these three cells carry no `+top{M}` cut, so no ranking cut can lose a
node: the pool is the walk, which already sits at 2,337 to 2,487 nodes). No U gold lies in the top-1000 dense or SPLADE lists either. **The
shortfall is not the reach** (PL2 10.5: the uncapped three-hop FULL walk reaches 1.000 / 0.998 / 0.996 of the missing golds on MuSiQue / MetaQA /
WebQSP): **it is the caps and the 2,500-node bound**. Holding the C golds costs nodes: the grid's own best ALL at any size is 0.9355 / 0.9653 /
0.8480 only at 5,783 / 6,268 / 6,067 nodes (2.3 to 2.5 times the bound), and MetaQA's 0.95 floor is reached at 4,984 nodes, 2.0 times the
bound. The next change has to attack **C**: which nodes the caps keep from each seed and each frontier, and how many nodes each hop may
spend, not the ranking and not the reach.

### 14.6 Beside the call: the e3v lists (post-hoc; decides nothing)

The call is read on the frozen lists as declared. The same R95 pair read on the e3v lists of the eval carve, with the call's own bootstrap and
status function (`outputs/pl3/e3v_beside.py`, record `outputs/pl3/call_e3v_beside.json`):

| dataset | NEW cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL difference, 95% CI | nodes ratio, 95% CI | status by the rule |
|---|---|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` | 0.9699 (50.0) | 0.9699 (50.0) | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| SQuAD | `rrf1000_b15` | 0.9421 (15.0) | 0.9421 (15.0) | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9642 (77.5) | 0.9646 (77.6) | -0.04 [-0.19, +0.08] | 0.9996 [0.9994, 0.9997] | ABOVE |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9684 (916.1) | 0.9750 (2,139.0) | -0.67 [-0.95, -0.40] | 0.4283 [0.4272, 0.4295] | ABOVE |
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8775 (2,457.0) | 0.8010 (1,061.1) | +7.65 [+6.58, +8.69] | 2.3155 [2.3034, 2.3272] | LEVEL |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.9296 (2,325.6) | 0.9281 (2,142.8) | +0.15 [+0.08, +0.24] | 1.0853 [1.0852, 1.0854] | LEVEL |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7618 (2,720.1) | 0.7432 (1,989.4) | +1.86 [-0.07, +3.73] | 1.3673 [1.3358, 1.3982] | LEVEL |

On e3v the status is the same on all seven datasets. **2Wiki** keeps its ABOVE by the price clause only: the new cell reads 0.9684 against PL2's 0.9750
(-0.67 points [-0.95, -0.40]) at 0.428 of the nodes; both reach 0.95. **HotpotQA** reads -0.04 points [-0.19, +0.08] at 0.9996 of the nodes. The three
SHORT datasets remain SHORT on e3v (MetaQA 0.9296, short by 2.0 points; MuSiQue 0.8775; WebQSP 0.7618 at 2,720 nodes). SQuAD's 15-node pool is 0.9421 on
e3v for both grids.

### 14.7 What the call means, and what it leaves unread

- **The construction rule.** Under CARRY_PL3 the own-construction rule of all seven datasets is the union grid's R95 cell (the table of 14.3).
  Against PL2's carried cells it changes five: HotpotQA (the same pool with `~cs`, 0.04 percent fewer nodes), 2Wiki (917 nodes instead of 2,140), MuSiQue
  (2,455 instead of 1,064: 2.31 times the nodes for +5.05 points), MetaQA (2,336 instead of 2,154) and WebQSP (2,468 instead of 1,891); MultiHop-RAG
  and SQuAD keep PL2's cells. This is the universal rule as UL2 states it (ABOVE somewhere, BELOW nowhere, one price bound, carried everywhere or nowhere):
  MuSiQue, MetaQA and WebQSP carry their larger cells because the rule is one rule, not because each is ABOVE.
- **What PL1's pool rows read.** PL1 section 7 item 4: a carried form carries to every arm at once and the frozen arms are rebuilt with it and graded again.
  The pool construction PL1's rows read is now PL3's R95 cell per dataset (names in 14.3; the dev picks are `outputs/pl3/<d>/select.json`, `picks.R95_all`).
  Pools built on PL2's cells for HotpotQA, 2Wiki, MuSiQue, MetaQA and WebQSP are built on the superseded construction. Whether an arm is frozen is
  PL1's freeze test (a candidate list empty), not this page's: this round's candidate list for the three SHORT datasets is the names of 14.8; for the four
  datasets that meet 0.95 it does not name a change.
- **The three SHORT datasets are SHORT under the carried rule** (eval 0.879, 0.897, 0.780 against 0.95); the call is about the construction rule, not
  the coverage target, and the target is not met there.
- **The declared rule cannot credit a gain bought with more nodes on a SHORT dataset.** MuSiQue's +5.05 points [+4.18, +5.92] at 2.31 times the nodes, and
  WebQSP's +1.66 at 1.31, are LEVEL by the rule; whether such a gain should count is a change of the rule and needs its own declaration before any further
  number. This round does not make it.
- **Two e3v facts a pool row needs.** (1) WebQSP's R95 cell reads 2,720 nodes on the e3v lists (the bound is read on the selection carve's frozen lists: 2,442 on
  dev, 2,468 on eval frozen); PL2's cell read 1,989 there. (2) 2Wiki's PL3 cell costs 0.67 points of ALL on e3v against PL2's cell and saves 57 percent of the
  nodes.

### 14.8 The next changes (names only; none is adopted, none is in this round's grid; each is declared on its own page before any number)

Against C, which nodes the caps keep: label-ordered caps (per-link rule labels ordered by the ledger's precision, on a UL2 layout), support-ordered
expansion (the number of seeds that reach a node), reading the per-link weights. Against C, how many nodes each hop may spend: a hop-3 frontier restricted
to the nodes the hop-2 ranking kept, per-family visited budgets. The rule's limit in 14.7 is a separate question for its own declaration.

### 14.9 The tables (the report, as of seven of seven read)

#### Gates and cost

| dataset | G-R (PL2's 804 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q x 804 cells = 180,096 elements, mismatches 0/0) | 3, 46 s, 0.28 GB |
| squad | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 2,032 s, 0.33 GB |
| hotpotqa | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 3,228 s, 14.84 GB |
| 2wiki | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,819 s, 17.09 GB |
| musique | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,054 s, 0.63 GB |
| metaqa | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 1,982 s, 0.39 GB |
| webqsp | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,122 s, 5.64 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 432, squad PASS 432; G-X3 PASS, PASS); code-path sha256 `e9d86095...`.

#### Dev reading (the rule's own numbers, dev carve, frozen lists): PL2's chosen cell against the union grid's

| dataset | dev q | PL2 R95 | PL3 R95 (all cells) | PL2 R98 | PL3 R98 | PL2's pick re-read = its filed pick |
|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b60` (0.9866 at 60.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | True |
| hotpotqa | 1508 | `rrf1000_b25+STRUCT:h1_c100+top100` (0.9615 at 77.3) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b100+STRUCT:h1_c100+top200` (0.9808 at 153.5) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | True |
| 2wiki | 1496 | `equal_rrf_budget_200+FULL:h2_c25` (0.9492 at 2,129.4, SHORT) | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `equal_rrf_budget_200+FULL:h2_c25` (0.9492 at 2,129.4, SHORT) | `rrf1000_b1600+FULL:h1_c100~s20+top800` (0.9606 at 2,078.2, SHORT) | True |
| musique | 1534 | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | True |
| metaqa | 1497 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | True |

#### The chosen cell at 0.95 (R95) and its confirmation on eval: PL2's against the union grid's

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL2 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag PL3 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| squad PL2 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT |
| squad PL3 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT |
| hotpotqa PL2 | `rrf1000_b25+STRUCT:h1_c100+top100` | 0.9615 | 77.3 | 0.9627 (77.9) | 0.9646 (77.6) |
| hotpotqa PL3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| 2wiki PL2 | `equal_rrf_budget_200+FULL:h2_c25` (SHORT) | 0.9492 | 2,129.4 | 0.9503 (2,140.1) | 0.9750 (2,139.0) |
| 2wiki PL3 | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| musique PL2 | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | 0.8010 (1,061.1) SHORT |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | 0.8775 (2,457.0) SHORT |
| metaqa PL2 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | 0.9281 (2,142.8) SHORT |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| webqsp PL2 | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | 0.7432 (1,989.4) SHORT |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT |

#### The chosen cell at 0.98 (R98) and its confirmation on eval: PL2's against the union grid's

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL2 | `rrf1000_b60` | 0.9866 | 60.0 | 0.9681 (60.0) SHORT | 0.9664 (60.0) SHORT |
| multihoprag PL3 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| squad PL2 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| squad PL3 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| hotpotqa PL2 | `rrf1000_b100+STRUCT:h1_c100+top200` | 0.9808 | 153.5 | 0.9776 (154.5) SHORT | 0.9743 (154.4) SHORT |
| hotpotqa PL3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| 2wiki PL2 | `equal_rrf_budget_200+FULL:h2_c25` (SHORT) | 0.9492 | 2,129.4 | 0.9503 (2,140.1) SHORT | 0.9750 (2,139.0) SHORT |
| 2wiki PL3 | `rrf1000_b1600+FULL:h1_c100~s20+top800` (SHORT) | 0.9606 | 2,078.2 | 0.9656 (2,085.7) SHORT | 0.9716 (2,088.6) SHORT |
| musique PL2 | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | 0.8010 (1,061.1) SHORT |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | 0.8775 (2,457.0) SHORT |
| metaqa PL2 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | 0.9281 (2,142.8) SHORT |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| webqsp PL2 | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | 0.7432 (1,989.4) SHORT |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT |

#### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | 0.9736 (50.0) |
| hotpotqa | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9684 (93.7) | 0.9646 (92.2) |
| 2wiki | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9058 (105.6) | 0.9310 (101.6) |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | 0.7911 (2,087.2) |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | 0.9278 (2,005.1) |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | 0.6447 (2,124.5) |

#### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL2's 804 cells / the union grid

| dataset | 0.95: PL2 | 0.95: PL3 | 0.98: PL2 | 0.98: PL3 |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b60` 0.9866 at 60.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100+top100` 0.9615 at 77.3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b100+STRUCT:h1_c100+top200` 0.9808 at 153.5 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 |
| 2wiki | `rrf1000_b800+FULL:h2_c25` 0.9545 at 2,695.6 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | none (best 0.9552 at 4,073.5) | none (best 0.9719 at 6,230.3) |
| musique | none (best 0.8449 at 2,611.7) | none (best 0.9355 at 5,783.4) | none (best 0.8449 at 2,611.7) | none (best 0.9355 at 5,783.4) |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best 0.9613 at 5,003.7) | none (best 0.9653 at 6,268.2) |
| webqsp | none (best 0.8016 at 5,100.2) | none (best 0.8480 at 6,067.3) | none (best 0.8016 at 5,100.2) | none (best 0.8480 at 6,067.3) |

#### The highest ALL on dev (the bound is 2,500 mean nodes): PL2's 804 cells / the union grid, within the bound and at any size

| dataset | PL2, within the bound | PL3, within the bound | PL2, any size | PL3, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| hotpotqa | `rrf1000_b800+FULL:h1_c100` 0.9927 at 935.0 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+FULL:h1_c100` 0.9927 at 935.0 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 |
| 2wiki | `equal_rrf_budget_200+FULL:h2_c25` 0.9492 at 2,129.4 | `rrf1000_b1600+FULL:h1_c100~s20+top800` 0.9606 at 2,078.2 | `equal_rrf_budget_200+FULL:h2_c25_v5000` 0.9552 at 4,073.5 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9719 at 6,230.3 |
| musique | `rrf1000_b800+FULL:h1_c100` 0.8012 at 1,082.6 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8735 at 2,488.9 | `rrf1000_b800+FULL:h2_c25` 0.8449 at 2,611.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8958 at 2,157.6 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v2000` 0.7505 at 1,888.2 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+FULL:h3_c100_v5000` 0.8016 at 5,100.2 | `rrf1000_b1600+FULL:h3_c100_v5000~vr` 0.8480 at 6,067.3 |

#### What PL3's cells bought (dev frontier of the union grid minus PL2's frontier, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9846 / 0.9846 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9920 / 0.9920 (+0.00) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9953 / 0.9960 (+0.07) | 0.9960 / 0.9967 (+0.07) | 0.9973 / 0.9973 (+0.00) | 0.9973 / 0.9987 (+0.13) | 0.9980 / 0.9993 (+0.13) | 0.9980 / 1.0000 (+0.20) |
| hotpotqa | 0.6127 / 0.6127 (+0.00) | 0.9224 / 0.9224 (+0.00) | 0.9496 / 0.9496 (+0.00) | 0.9682 / 0.9682 (+0.00) | 0.9775 / 0.9788 (+0.13) | 0.9808 / 0.9814 (+0.07) | 0.9828 / 0.9834 (+0.07) | 0.9834 / 0.9834 (+0.00) | 0.9834 / 0.9887 (+0.53) | 0.9834 / 0.9901 (+0.66) | 0.9927 / 0.9927 (+0.00) | 0.9927 / 0.9934 (+0.07) | 0.9927 / 0.9940 (+0.13) | 0.9927 / 0.9960 (+0.33) |
| 2wiki | 0.3195 / 0.3195 (+0.00) | 0.7172 / 0.7172 (+0.00) | 0.8342 / 0.8342 (+0.00) | 0.8890 / 0.8890 (+0.00) | 0.8984 / 0.8984 (+0.00) | 0.9078 / 0.9078 (+0.00) | 0.9164 / 0.9164 (+0.00) | 0.9298 / 0.9298 (+0.00) | 0.9325 / 0.9325 (+0.00) | 0.9325 / 0.9452 (+1.27) | 0.9358 / 0.9552 (+1.94) | 0.9378 / 0.9552 (+1.74) | 0.9378 / 0.9552 (+1.74) | 0.9492 / 0.9606 (+1.14) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6271 / 0.6349 (+0.78) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7066 / 0.7373 (+3.06) | 0.7151 / 0.7673 (+5.22) | 0.7881 / 0.7947 (+0.65) | 0.8012 / 0.8246 (+2.35) | 0.8012 / 0.8449 (+4.37) | 0.8012 / 0.8735 (+7.24) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7214 / 0.7308 (+0.94) | 0.7408 / 0.8938 (+15.30) | 0.8958 / 0.8978 (+0.20) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6025 / 0.6058 (+0.33) | 0.6251 / 0.6603 (+3.52) | 0.6251 / 0.6802 (+5.51) | 0.6251 / 0.6802 (+5.51) | 0.6251 / 0.6855 (+6.04) | 0.6676 / 0.7147 (+4.71) | 0.6702 / 0.7339 (+6.37) | 0.7505 / 0.7571 (+0.66) | 0.7505 / 0.7731 (+2.26) |

(each entry: PL2's frontier / the union grid's frontier (points bought); `-` = no cell is that small)

#### What each family bought: R95 of PL2's cells plus that family's cells (dev), and the best ALL inside the bound among that family's cells alone

| dataset | PL2 R95 | deeper base | similarity caps | rank cap | both | FS regime | seeds 10/20 | bridge |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 123.2 | = PL2; alone best 1.0000 at 111.5 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 200.0 |
| squad | 0.9519 at 15.0 | = PL2; alone best 0.9987 at 1,928.5 | = PL2; alone best 1.0000 at 2,190.4 | = PL2; alone best 0.9987 at 1,928.5 | = PL2; alone best 1.0000 at 2,190.4 | = PL2; alone best 0.9993 at 1,744.6 | = PL2; alone best 0.9993 at 1,944.7 | = PL2; alone best 0.9973 at 795.7 |
| hotpotqa | 0.9615 at 77.3 | = PL2; alone best 0.9947 at 2,323.3 | 0.9622 at 77.3; alone best 0.9960 at 2,246.8 | = PL2; alone best 0.9947 at 2,323.3 | = PL2; alone best 0.9960 at 2,246.8 | = PL2; alone best 0.9934 at 2,439.1 | = PL2; alone best 0.9940 at 1,880.9 | = PL2; alone best 0.9881 at 542.8 |
| 2wiki | 0.9492 at 2,129.4 SHORT | 0.9532 at 2,350.9; alone best 0.9532 at 2,350.9 | = PL2; alone best 0.9432 at 1,799.3 | 0.9559 at 2,334.0; alone best 0.9559 at 2,334.0 | 0.9525 at 2,311.9; alone best 0.9525 at 2,311.9 | 0.9539 at 2,000.0; alone best 0.9572 at 2,358.1 | 0.9545 at 908.2; alone best 0.9606 at 2,078.2 | = PL2; alone best 0.9358 at 614.1 |
| musique | 0.8012 at 1,082.6 SHORT | 0.8605 at 2,386.3 SHORT; alone best 0.8605 at 2,386.3 | 0.8611 at 2,394.6 SHORT; alone best 0.8611 at 2,394.6 | 0.8527 at 2,400.0 SHORT; alone best 0.8527 at 2,400.0 | 0.8735 at 2,488.9 SHORT; alone best 0.8735 at 2,488.9 | 0.8546 at 2,419.9 SHORT; alone best 0.8546 at 2,419.9 | 0.8566 at 2,144.6 SHORT; alone best 0.8566 at 2,144.6 | = PL2; alone best 0.7608 at 677.1 |
| metaqa | 0.8958 at 2,157.6 SHORT | 0.8978 at 2,339.7 SHORT; alone best 0.8978 at 2,339.7 | = PL2; alone best 0.7522 at 2,410.0 | = PL2; alone best 0.7548 at 2,465.9 | = PL2; alone best 0.7522 at 2,410.0 | = PL2; alone best 0.8904 at 2,335.5 | = PL2; alone best 0.7428 at 2,286.6 | = PL2; alone best 0.3621 at 452.7 |
| webqsp | 0.7505 at 1,888.2 SHORT | 0.7532 at 2,246.5 SHORT; alone best 0.7532 at 2,246.5 | 0.7624 at 2,289.1 SHORT; alone best 0.7624 at 2,289.1 | 0.7545 at 2,246.5 SHORT; alone best 0.7545 at 2,246.5 | 0.7624 at 2,288.5 SHORT; alone best 0.7624 at 2,288.5 | 0.7724 at 2,367.7 SHORT; alone best 0.7724 at 2,367.7 | 0.7731 at 2,442.4 SHORT; alone best 0.7731 at 2,442.4 | = PL2; alone best 0.6317 at 479.9 |

#### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

#### The call: CARRY_PL3 (read 7 of 7: 2wiki, hotpotqa, metaqa, multihoprag, musique, squad, webqsp)

| dataset | status | NEW (PL3 R95) | NEW ALL (nodes) | OLD (PL2 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | ABOVE | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100+top100` | 0.9627 (77.9) | +0.07 [-0.07, +0.19] | 1.000 [0.999, 1.000] |
| 2wiki | ABOVE | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `equal_rrf_budget_200+FULL:h2_c25` | 0.9503 (2,140.1) | +1.06 [+0.74, +1.41] | 0.428 [0.427, 0.430] |
| musique | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) | `rrf1000_b800+FULL:h1_c100` | 0.8283 (1,064.2) | +5.05 [+4.18, +5.92] | 2.307 [2.295, 2.320] |
| metaqa | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` | 0.8952 (2,153.5) | +0.15 [+0.07, +0.24] | 1.085 [1.085, 1.085] |
| webqsp | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | `dense_top200+FULL:h2_c100_v2000` | 0.7638 (1,890.8) | +1.66 [-0.07, +3.39] | 1.305 [1.276, 1.333] |

Beside the call (decide nothing): R98 and the `new`-only pick against PL2's R95.

| dataset | R98 status | R98 NEW ALL (nodes) vs PL2 | new-only status | new-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | ABOVE | 0.9788 (58.0) vs 0.9681 (60.0) | LEVEL | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-vr+top50` | 0.9773 (72.8) |
| hotpotqa | ABOVE | 0.9783 (153.5) vs 0.9776 (154.5) | ABOVE | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) |
| 2wiki | ABOVE | 0.9656 (2,085.7) vs 0.9503 (2,140.1) | ABOVE | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) |
| musique | LEVEL | 0.8788 (2,455.1) vs 0.8283 (1,064.2) | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8952 (2,153.5) | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) |
| webqsp | LEVEL | 0.7804 (2,468.1) vs 0.7638 (1,890.8) | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) |

#### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 300 | 36 (0.120) | 38 | 1.06 | 0 (q 0) | 38 (q 36) | 0 (q 0) | 0 | 2,487.2 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 300 | 63 (0.210) | 245 | 3.89 | 0 (q 0) | 245 (q 63) | 0 (q 0) | 0 | 2,421.1 |

### 14.10 Surprises and contradictions with filed numbers

- PL2's section 10.5 pointed at similarity-ordered caps as the change aimed at 2Wiki. They bought nothing there (PL2's cells plus `cs` = PL2); **seed widening
  (`s20`) did**, at a smaller price. Similarity-ordered caps bought only HotpotQA's 0.07 points at the same nodes (the call's thin ABOVE) and parts of MuSiQue
  and WebQSP.
- 2Wiki's PL2 pick is SHORT on dev (0.9492) but reaches 0.95 on eval frozen (0.9503) and 0.975 on e3v; PL3's reaches it on dev, eval and e3v.
- The e3v lists lift MetaQA by 3.3 points over the frozen lists (0.9296 against 0.8968) and lower WebQSP and SQuAD.
- The three SHORT datasets did not move to 0.95 in any family or any combination inside the bound; the largest grid cell reaches 0.9355, 0.9653 and 0.8480 only at 2.3
  to 2.5 times the bound.

### 14.11 Disclosed and undone

- D1 was run on 300 dev questions chosen evenly over the carve (`np.linspace`), not on a prefix: the first run took the first 200 questions of MetaQA, whose
  carve is ordered by hop count, and found that only 2 of them missed a gold; it was discarded and the sampling rule changed to the even spacing before D1's
  numbers were filed. That change to `zpl3r.py` (the report and diagnostic module, not the engine) is committed at 8bf1acf. D1's labels are post-hoc and decide nothing.
- The dev-frontier readings in 14.3 are of dev only, over the 2,274 cells of the union: optimistic by selection, and the call does not read them. They are stated
  because the declared rule cannot see a gain bought at more nodes on a SHORT dataset (14.7).
- The e3v reading of 14.6 (`outputs/pl3/e3v_beside.py`) was written after the call was read; it uses the call's own bootstrap and status function and decides nothing.
- The options `cs`, `vr`, `s10`, `s20`, the `FS` regime and `brg` have no counterpart in `m3b_pools`, so G-R cannot test them; their checks are G-X3's independent
  per-node loop and the self-test's brute-force pool builder (both PASS, `selftest.json`).
- The laptop chain was restarted once, for process reasons only (a serial frozen confirmation of SQuAD blocked the next dataset's grid); no number had been read that
  the restart could have changed. The host rows ran as filed in `outputs/pl3/feed_rows.txt` under the coordinator's classes.
- The call's bootstrap draws from one generator across the datasets in the order of the call; when HotpotQA and 2Wiki entered, the draws for WebQSP
  shifted, so WebQSP's ALL-difference interval is [-0.07, +3.39] here and was [-0.33, +3.53] in the five-of-seven filing at 8bf1acf (the nodes ratio is the same
  to three digits). Its status is LEVEL in both (the nodes ratio's interval lies above 1, so ABOVE clauses 2 and 3 cannot fire, and the ALL interval's upper end is
  above 0, so BELOW clause 2 cannot). MuSiQue's, MetaQA's, SQuAD's and MultiHop-RAG's intervals are unchanged.
- The grid arrays (`*.npz`) are not committed (as PL2's); the heads, selects, confirmations, the call and the report are.
- Not done: the eval confirmation of the frontier cells below the bound on the SHORT datasets; the next changes of 14.8; PL1's rebuild of the pools on the carried
  construction (PL1's rows).
