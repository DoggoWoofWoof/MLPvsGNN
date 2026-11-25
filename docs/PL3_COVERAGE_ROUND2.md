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
