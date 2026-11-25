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

## 14. Results, filed 25 Nov 2026: five of seven read (the five laptop datasets); HotpotQA and 2Wiki are host rows, not read

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl3/<d>/select.json`, `confirm_frozen.json`,
`confirm_e3v.json` for MultiHop-RAG, `outputs/pl3/call.json`) by `outputs/mp_unified/zpl3r.py report`; the tables of section 14.7 are
pasted unchanged from `outputs/pl3/report.md`. The engine is the one committed at 7a55929 (`zpl3g.py` is byte-identical to the
declaration commit; its sha256 starts `20ad63a08f5d` in all five select heads, the cell list `ff8798810909`, the code-path hash
`e9d860950c70` is equal on the five datasets, G-U PASS). G-R (the engine rebuilds PL2's 804 cells element for element) PASSED on the five,
with 0 mismatches on 400 dev questions per dataset (224 on MultiHop-RAG) as filed in `gr_<d>.json`; the self-test (G-X3, brute-force
pools, G-F, G-D) passed. Two of the seven datasets are not read, so the call below is **INCOMPLETE**.

### 14.1 The call: INCOMPLETE (read 5 of 7: BELOW 0, ABOVE 0, LEVEL 5)

| dataset | status | NEW (PL3 R95) | OLD (PL2 R95) | ALL difference, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|
| MultiHop-RAG | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | the same | +0.00 | 1.000 |
| SQuAD | LEVEL (same cell) | `rrf1000_b15` 0.9554 at 15.0 | the same | +0.00 | 1.000 |
| MuSiQue (SHORT) | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8788 at 2,455.1 | `rrf1000_b800+FULL:h1_c100` 0.8283 at 1,064.2 | +5.05 [+4.18, +5.92] | 2.307 [2.295, 2.320] |
| MetaQA (SHORT) | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8952 at 2,153.5 | +0.15 [+0.07, +0.24] | 1.085 [1.085, 1.085] |
| WebQSP (SHORT) | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7804 at 2,468.1 | `dense_top200+FULL:h2_c100_v2000` 0.7638 at 1,890.8 | +1.66 [-0.33, +3.53] | 1.305 [1.276, 1.333] |
| HotpotQA | NOT READ | | | | |
| 2Wiki | NOT READ | | | | |

(eval, frozen lists; the whole table with the R98 and `new`-only readings beside it is in 14.7.) Under the declared rule the call can
become CARRY_PL3 only if HotpotQA or 2Wiki is ABOVE at R95 and neither is BELOW: the five read datasets are all LEVEL, and every one of
their chosen cells is within the 2,500 bound.

**A limit of the declared rule, stated here and not changed.** On a SHORT dataset (no cell reaches 0.95 inside the bound) R95 is the
highest ALL, and ABOVE needs the nodes ratio's upper end below 1 or the ALL difference's lower end above 0 *with the nodes ratio at or
below 1*. A cell that buys ALL with more nodes is LEVEL however large the gain: MuSiQue's pick is +5.05 points of ALL [+4.18, +5.92] on
eval for 2.31x the nodes, and the rule calls that LEVEL. Whether such a gain should count is a change of the rule and needs its own
declaration before any further number; this round does not make it.

### 14.2 Per dataset: PL2's chosen cell against the new frontier (dev = the selection carve, eval = the confirmation carve)

- **MultiHop-RAG** (224 dev questions; e3v read). R95: the new cells do not beat PL2's `rrf1000_b25+FULL:h1_c100+top25` (dev 0.9777 at
  50.0; eval 0.9690 frozen, 0.9699 e3v at 50.0 nodes). R98: PL2's `rrf1000_b60` (dev 0.9866 at 60.0; eval 0.9681 frozen / 0.9664 e3v,
  short of 0.98 by 1.2 / 1.4 points) against PL3's `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (dev 0.9955 at 58.6; eval 0.9788 frozen /
  0.9779 e3v at 58.0 / 58.2 nodes, short of 0.98 by 0.12 / 0.21 points): the one ABOVE in the five, beside the call (R98 decides
  nothing). Highest ALL: 1.0000 at 100.0 nodes (PL2's `rrf1000_b50+NER:h3_c100_v5000+top50`), the same in both grids.
- **SQuAD** (1,498 dev questions). R95 and R98 are both PL2's cells (`rrf1000_b15`: dev 0.9519 at 15.0, eval 0.9554; and
  `rrf1000_b50+KNN:h1_c100+top25`: dev 0.9806 at 63.3, eval 0.9818 at 63.0): nothing new is smaller at either floor. The new cells buy
  only at large size: highest ALL inside the bound 1.0000 at 2,190.4 nodes (`rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800`) against
  PL2's 0.9980 at 1,544.8.
- **MuSiQue** (1,534 dev questions; SHORT). PL2: `rrf1000_b800+FULL:h1_c100`, dev 0.8012 at 1,082.6, eval 0.8283 at 1,064.2, short of
  0.95 by 12.17 points. PL3: `rrf1000_b800+FULL:h3_c25_v2000~cs-vr`, dev 0.8735 at 2,488.9, eval 0.8788 at 2,455.1, short by 7.12
  points. Smallest cell at 0.95 at any size: none; the highest ALL at any size is 0.9355 at 5,783.4 nodes
  (`rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr`), i.e. 2.3 times the bound and still below 0.95; at 0.98: none. Dev frontier at equal
  nodes (a reading of the filed tables, dev only): +3.06 points at 500 nodes, +5.22 at 750, +2.35 at 1,500, +4.37 at 2,000, +7.24 at
  2,500.
- **MetaQA** (1,497 dev questions; SHORT). PL2: `equal_rrf_budget_200+STRUCT:h3_c25_v2000`, dev 0.8958 at 2,157.6, eval 0.8952 at
  2,153.5, short by 5.48 points. PL3: `rrf1000_b400+STRUCT:h3_c25_v2000`, dev 0.8978 at 2,339.7, eval 0.8968 at 2,335.9, short by 5.32
  points. Smallest cell at 0.95 at any size: PL2 `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 nodes; PL3
  `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6, about twice the bound; at 0.98: none (best 0.9653 at 6,268.2). The frontier's
  +15.30 points at 2,000 nodes is a lattice artifact, not a mechanism: PL2's frontier has no cell between 1,936.7 and 2,157.6 nodes,
  and PL3's `rrf1000_b25+STRUCT:h3_c25_v2000` (PL2's walk on a smaller base) reads 0.8938 at 1,999.3 against PL2's 0.8958 at 2,157.6.
- **WebQSP** (1,507 dev questions; SHORT). PL2: `dense_top200+FULL:h2_c100_v2000`, dev 0.7505 at 1,888.2, eval 0.7638 at 1,890.8, short
  by 18.62 points. PL3: `dense_top200+FULL:h2_c100_v5000~s10-cs-vr`, dev 0.7731 at 2,442.4, eval 0.7804 at 2,468.1, short by 16.96
  points. Smallest cell at 0.95 at any size: none; highest ALL at any size 0.8480 at 6,067.3 nodes
  (`rrf1000_b1600+FULL:h3_c100_v5000~vr`). Dev frontier at equal nodes: +3.52 points at 300 nodes, +5.51 at 400 and 500, +6.04 at
  750, +4.71 at 1,000, +6.37 at 1,500, +0.66 at 2,000, +2.26 at 2,500.
- **HotpotQA, 2Wiki**: not read. Their `pl3-*` rows are in `outputs/pl3/feed_rows.txt` (14.6).

Selection optimism, measured: the SHORT datasets' picks moved from dev to eval by +0.53 points (MuSiQue), +0.73 (WebQSP) and -0.10
(MetaQA); MultiHop-RAG's R95 cell by -0.87 (224 dev questions); SQuAD's by +0.35.

### 14.3 What each family bought (R95 of PL2's cells plus that family, dev; the best ALL inside the bound of that family alone)

- **MultiHop-RAG, SQuAD**: no family moves R95 (the floor is reached at PL2's smallest cell); the rank cap, similarity caps and the
  deeper base all give 1.0000 or 0.9987 alone at 125 to 2,190 nodes.
- **MuSiQue** (PL2 0.8012 at 1,082.6): deeper base 0.8605 at 2,386.3; similarity caps 0.8611 at 2,394.6; visited cap by rank 0.8527 at
  2,400.0; both caps together 0.8735 at 2,488.9 (the largest step); FS regime 0.8546 at 2,419.9; seed widening 0.8566 at 2,144.6;
  bridge-conditioned ranking: nothing (equal to PL2; alone best 0.7608 at 677.1).
- **MetaQA** (PL2 0.8958 at 2,157.6): only a deeper base moves it (0.8978 at 2,339.7); similarity caps, the rank cap, FS regime, seeds
  and bridge do nothing inside the bound (alone: 0.7522, 0.7548, 0.8904, 0.7428, 0.3621).
- **WebQSP** (PL2 0.7505 at 1,888.2): seed widening 0.7731 at 2,442.4 (the largest step; it is the picked cell's `s10`); FS regime
  0.7724 at 2,367.7; similarity caps 0.7624 at 2,289.1 (both caps together the same); the rank cap 0.7545; deeper base 0.7532;
  bridge: nothing (alone best 0.6317 at 479.9).
- **Bridge-conditioned ranking bought nothing on any of the five** (it only re-ranks the nodes the walk reached; D1's R class below is 0).
- No family brings any SHORT dataset to 0.95 inside the bound. The best reading of R95 moves dev ALL by +0.20 (MetaQA), +2.26
  (WebQSP) and +7.23 (MuSiQue) points, at 2,339.7, 2,442.4 and 2,488.9 mean nodes.

### 14.4 What is still short, in numbers (diagnostic D1, post-hoc, labels read; 300 dev questions evenly spaced over the carve)

Each gold node missing from the pool of the chosen R95 cell has one class: **R** ranked past M, **C** not in the cell's own walk but in
the uncapped three-hop FULL reach of its seeds (the caps or the order dropped it), **U** in neither (the reach ceiling).

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (bound 2,500) |
|---|---|---|---|---|---|---|---|
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 36 of 300 (0.120) | 38 (1.06 per such question) | 0 | 38 | 0 | 2,487.2 |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | 33 of 300 (0.110) | 375 (11.36 per such question) | 0 | 373 | 2 | 2,337.4 |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 63 of 300 (0.210) | 245 (3.89 per such question) | 0 | 245 | 0 | 2,421.1 |

Of the 658 missing golds, 656 (99.7%) are C and 2 are U; none is R (these three cells carry no `+top{M}` cut, so no ranking cut can
lose a node: the pool is the walk, which already sits at 2,337 to 2,487 nodes). No U gold lies in the top-1000 dense or SPLADE lists
either. **The shortfall is not the reach** (PL2 10.5: the uncapped three-hop FULL walk reaches 1.000 / 0.998 / 0.996 of the missing
golds on MuSiQue / MetaQA / WebQSP): **it is the caps and the 2,500-node bound**. Holding the C golds costs nodes: the grid's own best
ALL at any size is 0.9355 / 0.9653 / 0.8480 only at 5,783 / 6,268 / 6,067 nodes (2.3 to 2.5 times the bound), and MetaQA's 0.95 floor
is reached at 4,984 nodes, 2.0 times the bound. The next change has to attack **C**: which nodes the caps keep from each seed and each
frontier, and how many nodes each hop may spend, not the ranking and not the reach.

### 14.5 The next changes (names only; none is adopted, none is in this round's grid; each is declared on its own page before any number)

Against C, which nodes the caps keep: label-ordered caps (per-link rule labels ordered by the ledger's precision, on a UL2 layout),
support-ordered expansion (the number of seeds that reach a node), reading the per-link weights. Against C, how many nodes each hop
may spend: a hop-3 frontier restricted to the nodes the hop-2 ranking kept, per-family visited budgets. The rule's limit stated in
14.1 (a gain bought with more nodes on a SHORT dataset) is a separate question for its own declaration.

### 14.6 Host rows still to be read, and what the call waits for

`outputs/pl3/feed_rows.txt` (19 rows, not sent by this agent; the coordinator remaps classes and dependencies at insertion):
`pl3-gr-musique` (the host's G-R, on MuSiQue; the first row, the other grid rows depend on it); for HotpotQA and 2Wiki
`pl3-gr-<d>`, the dev grid in three parts `pl3-g-<d>-0..2`, `pl3-sel-<d>`, `pl3-cf-<d>` (eval, frozen lists), `pl3-ce-<d>` (eval, e3v
lists); and `pl3-ce-{squad,musique,metaqa,webqsp}` (the e3v confirmations of the four laptop datasets, whose vectors sit on the host;
their input is `outputs/pl3/<d>/select.json`, filed in the commit of this section). The grid rows read PL2's dev grids
(`pl2-*` rows) for HotpotQA and 2Wiki; `pl3-gr-musique` reads PL2's laptop-only MuSiQue grid, which needs `rx push --inputs`. After
`pl3-sel-hotpotqa` and `pl3-sel-2wiki`, `zpl3g.py call` and `zpl3r.py report` re-read the seven datasets; the result is a new section
here, not an edit of this one.

### 14.7 The tables (the report, as of five of seven read)

#### Gates and cost

| dataset | G-R (PL2's 804 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q x 804 cells = 180,096 elements, mismatches 0/0) | 3, 46 s, 0.28 GB |
| squad | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 2,032 s, 0.33 GB |
| hotpotqa | not run | NOT READ |
| 2wiki | not run | NOT READ |
| musique | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,054 s, 0.63 GB |
| metaqa | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 1,982 s, 0.39 GB |
| webqsp | PASS (400 q x 804 cells = 321,600 elements, mismatches 0/0) | 3, 4,122 s, 5.64 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 432, squad PASS 432; G-X3 PASS, PASS); code-path sha256 `e9d86095...`.

#### Dev reading (the rule's own numbers, dev carve, frozen lists): PL2's chosen cell against the union grid's

| dataset | dev q | PL2 R95 | PL3 R95 (all cells) | PL2 R98 | PL3 R98 | PL2's pick re-read = its filed pick |
|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b60` (0.9866 at 60.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | True |
| hotpotqa | NOT READ | | | | | |
| 2wiki | NOT READ | | | | | |
| musique | 1534 | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | True |
| metaqa | 1497 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | True |

#### The chosen cell at 0.95 (R95) and its confirmation on eval: PL2's against the union grid's

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL2 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag PL3 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| squad PL2 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| squad PL3 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| musique PL2 | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | not read |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | not read |
| metaqa PL2 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | not read |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| webqsp PL2 | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | not read |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |

#### The chosen cell at 0.98 (R98) and its confirmation on eval: PL2's against the union grid's

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL2 | `rrf1000_b60` | 0.9866 | 60.0 | 0.9681 (60.0) SHORT | 0.9664 (60.0) SHORT |
| multihoprag PL3 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| squad PL2 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| squad PL3 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| musique PL2 | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | not read |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | not read |
| metaqa PL2 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | not read |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| webqsp PL2 | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | not read |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |

#### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | not read |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | not read |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | not read |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | not read |

#### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL2's 804 cells / the union grid

| dataset | 0.95: PL2 | 0.95: PL3 | 0.98: PL2 | 0.98: PL3 |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b60` 0.9866 at 60.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| musique | none (best 0.8449 at 2,611.7) | none (best 0.9355 at 5,783.4) | none (best 0.8449 at 2,611.7) | none (best 0.9355 at 5,783.4) |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best 0.9613 at 5,003.7) | none (best 0.9653 at 6,268.2) |
| webqsp | none (best 0.8016 at 5,100.2) | none (best 0.8480 at 6,067.3) | none (best 0.8016 at 5,100.2) | none (best 0.8480 at 6,067.3) |

#### The highest ALL on dev (the bound is 2,500 mean nodes): PL2's 804 cells / the union grid, within the bound and at any size

| dataset | PL2, within the bound | PL3, within the bound | PL2, any size | PL3, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| musique | `rrf1000_b800+FULL:h1_c100` 0.8012 at 1,082.6 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8735 at 2,488.9 | `rrf1000_b800+FULL:h2_c25` 0.8449 at 2,611.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8958 at 2,157.6 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v2000` 0.7505 at 1,888.2 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+FULL:h3_c100_v5000` 0.8016 at 5,100.2 | `rrf1000_b1600+FULL:h3_c100_v5000~vr` 0.8480 at 6,067.3 |

#### What PL3's cells bought (dev frontier of the union grid minus PL2's frontier, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9846 / 0.9846 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9920 / 0.9920 (+0.00) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9953 / 0.9960 (+0.07) | 0.9960 / 0.9967 (+0.07) | 0.9973 / 0.9973 (+0.00) | 0.9973 / 0.9987 (+0.13) | 0.9980 / 0.9993 (+0.13) | 0.9980 / 1.0000 (+0.20) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6271 / 0.6349 (+0.78) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7066 / 0.7373 (+3.06) | 0.7151 / 0.7673 (+5.22) | 0.7881 / 0.7947 (+0.65) | 0.8012 / 0.8246 (+2.35) | 0.8012 / 0.8449 (+4.37) | 0.8012 / 0.8735 (+7.24) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7214 / 0.7308 (+0.94) | 0.7408 / 0.8938 (+15.30) | 0.8958 / 0.8978 (+0.20) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6025 / 0.6058 (+0.33) | 0.6251 / 0.6603 (+3.52) | 0.6251 / 0.6802 (+5.51) | 0.6251 / 0.6802 (+5.51) | 0.6251 / 0.6855 (+6.04) | 0.6676 / 0.7147 (+4.71) | 0.6702 / 0.7339 (+6.37) | 0.7505 / 0.7571 (+0.66) | 0.7505 / 0.7731 (+2.26) |

(each entry: PL2's frontier / the union grid's frontier (points bought); `-` = no cell is that small)

#### What each family bought: R95 of PL2's cells plus that family's cells (dev), and the best ALL inside the bound among that family's cells alone

| dataset | PL2 R95 | deeper base | similarity caps | rank cap | both | FS regime | seeds 10/20 | bridge |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 123.2 | = PL2; alone best 1.0000 at 111.5 | = PL2; alone best 1.0000 at 125.0 | = PL2; alone best 1.0000 at 200.0 |
| squad | 0.9519 at 15.0 | = PL2; alone best 0.9987 at 1,928.5 | = PL2; alone best 1.0000 at 2,190.4 | = PL2; alone best 0.9987 at 1,928.5 | = PL2; alone best 1.0000 at 2,190.4 | = PL2; alone best 0.9993 at 1,744.6 | = PL2; alone best 0.9993 at 1,944.7 | = PL2; alone best 0.9973 at 795.7 |
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

#### The call: INCOMPLETE (read 5 of 7: metaqa, multihoprag, musique, squad, webqsp)

| dataset | status | NEW (PL3 R95) | NEW ALL (nodes) | OLD (PL2 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | NOT READ | | | | | | |
| 2wiki | NOT READ | | | | | | |
| musique | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) | `rrf1000_b800+FULL:h1_c100` | 0.8283 (1,064.2) | +5.05 [+4.18, +5.92] | 2.307 [2.295, 2.320] |
| metaqa | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` | 0.8952 (2,153.5) | +0.15 [+0.07, +0.24] | 1.085 [1.085, 1.085] |
| webqsp | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | `dense_top200+FULL:h2_c100_v2000` | 0.7638 (1,890.8) | +1.66 [-0.33, +3.53] | 1.305 [1.276, 1.333] |

Beside the call (decide nothing): R98 and the `new`-only pick against PL2's R95.

| dataset | R98 status | R98 NEW ALL (nodes) vs PL2 | new-only status | new-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | ABOVE | 0.9788 (58.0) vs 0.9681 (60.0) | LEVEL | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-vr+top50` | 0.9773 (72.8) |
| musique | LEVEL | 0.8788 (2,455.1) vs 0.8283 (1,064.2) | LEVEL | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8952 (2,153.5) | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) |
| webqsp | LEVEL | 0.7804 (2,468.1) vs 0.7638 (1,890.8) | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) |

#### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 300 | 36 (0.120) | 38 | 1.06 | 0 (q 0) | 38 (q 36) | 0 (q 0) | 0 | 2,487.2 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 300 | 63 (0.210) | 245 | 3.89 | 0 (q 0) | 245 (q 63) | 0 (q 0) | 0 | 2,421.1 |

### 14.8 Disclosed (this section)

- D1 was run on 300 dev questions chosen evenly over the carve (`np.linspace`), not on a prefix: the first run took the first 200 questions
  of MetaQA, whose carve is ordered by hop count, and found that only 2 of them missed a gold; it was discarded and the sampling rule
  changed to the even spacing before D1's numbers were filed. That change to `zpl3r.py` (the report and diagnostic module, not the
  engine) is committed with these records. D1's labels are post-hoc and decide nothing.
- The dev-frontier readings in 14.2 are of dev only, over the 2,274 cells of the union: optimistic by selection, and the call does not
  read them. They are stated because the declared rule cannot see a gain bought at more nodes on a SHORT dataset (14.1).
- e3v lists were read only for MultiHop-RAG; the other four datasets' e3v confirmations are host rows (14.6).
- The options `cs`, `vr`, `s10`, `s20`, the `FS` regime and `brg` have no counterpart in `m3b_pools`, so G-R cannot test them; their
  checks are G-X3's independent per-node loop and the self-test's brute-force pool builder (both PASS, `selftest.json`).
- The laptop chain was restarted once, for process reasons only (a serial frozen confirmation of SQuAD blocked the next dataset's grid);
  no number had been read that the restart could have changed.
