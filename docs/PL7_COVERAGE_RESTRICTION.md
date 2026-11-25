# PL7: the coverage target, round 6 (restricting the pool back toward 2,500 nodes, one rule on all seven, on the current links)

Declared 25 November 2026, before any PL7 coverage number exists. What I have read: PL6's declaration (744811e6) and its filed records: the dev selects of all seven datasets (`outputs/pl6/<d>/select.json`, rung 0 for five, the
ladders' last rungs for MuSiQue and WebQSP), the eval confirmations of MuSiQue and WebQSP (host, `grid_eval_frozen_confirm`) and of MetaQA, MultiHop-RAG, HotpotQA and 2Wiki (PL5's confirmation grids), and PL6's call over the six
datasets read so far (`outputs/pl6/call.json`, **preliminary: SQuAD's PL5 confirmation is still running, and PL6's results page is not yet written**). **No PL6 or PL5 record is changed by this page.** Before this commit I ran only label-free
or record-only things: the cell lists and their sha256 (`zpl7g.py cells`), the self-test (`outputs/pl7/selftest.json`: PASS in 56.6 s; hash equalities, parse checks, the engine against a per-node reference and the hard cuts' sizes on
1 real question each of MetaQA and MuSiQue; no coverage number), a code test of `select` for SQuAD and HotpotQA into a scratch folder (it re-reads PL6's table at K = 2,500, adds nothing; not kept) and a code test of gate G-C on arrays cut out
of PL5's filed grid (not kept). **Nothing here trains, no model is fitted, no LLM or spaCy is used, and every pool function is gold-free.** This page is the sixth round of the coverage target (docs/PL1_POOLS_ON_UL2_LAYOUT.md section 7).

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how much ever is possible".
- 25 Nov, about 15:00: asked whether the 2,500 mean-node bound may be lifted: "i do not mind just rules should be universal." PL6 (744811e6) lifted it under one rule (rule U).
- 25 Nov, after PL6's ladders had read (relayed by the coordinator): **"by lifting that bound i asked you to do analysis, we still need to start restricting it and bring towards 2.5k."** This page is the reading of that ruling:
  PL6 was analysis; the node count is brought back down toward 2,500 by ONE rule on all seven; **2,500 is where the ladder ends and the only place a carry is read**; a pick above 2,500 is a reading on the way down, never a final bound.
- 25 Nov, about 23:00, sequencing from the user (relayed): "we currently do not have proper gpu access ... so till then we need to properly finish the pools, encoding finish for all datasets, improving the R links, so that we can target the coverage budget and precision and
  eliminate the links we might not need, and keep the links we do actually need based on the gold links ... once that is done we will have corpus and the pools we can target the vector improvement and models after that with SEM2."
  **Consequence for this page, stated at its head: the pools it reads are on the CURRENT links, and they are PROVISIONAL until the link set is final** (section 10). This page is the restriction map toward 2,500 on the current links; its rule, its ladder and its calls carry to the
  tightened links, its cells, picks and knees do not.
- Earlier, still binding: "the pools or the rules or links all should be universal"; pools first and parameter-free (no learning, no model, no LLM, no spaCy); one model per family; the hub line is CLOSED (UL2 8.15); every rule is universal on all seven
  (MultiHop-RAG, MuSiQue, SQuAD, MetaQA, WebQSP, HotpotQA, 2Wiki), carried everywhere or nowhere.

## 1. What PL6's records say, and what I had seen before writing this page

Eval, frozen lists, the best cell by dev ALL within N mean dev nodes (PL6's matched-node reading, the cell chosen on dev and read once on eval; ALL @ mean eval nodes), against the bounded pick PL2 to PL5 carried:

| dataset | bounded pick (PL5) | N = 3,000 | 5,000 | 6,000 | 10,000 | 15,000 | 20,000 | 40,000 | PL6's U95 pick (eval) |
|---|---|---|---|---|---|---|---|---|---|
| musique | 0.8846 @ 2,438 | 0.8904 @ 2,950 | 0.8945 @ 2,980 | 0.9222 @ 5,742 | 0.9350 @ 9,076 | 0.9429 @ 13,615 | 0.9661 @ 16,576 | 0.9710 @ 20,236 | 0.9640 @ 15,400 |
| webqsp | 0.8789 @ 2,095 | 0.8802 @ 2,520 | 0.8975 @ 4,977 | 0.9168 @ 5,489 | 0.9175 @ 6,080 | 0.9328 @ 10,817 | 0.9328 @ 10,817 | 0.9474 @ 39,530 | 0.9474 @ 39,530 |
| metaqa | 0.8968 @ 2,336 | 0.8999 @ 2,699 | 0.9582 @ 4,973 | 0.9599 @ 5,602 | 0.9615 @ 6,259 | 0.9615 @ 6,259 | 0.9615 @ 6,259 | 0.9615 @ 6,259 | 0.9582 @ 4,973 |

(MetaQA's N = 4,000 reads 0.9244 @ 3,600; WebQSP's N = 30,000 reads 0.9421 @ 20,738 and N = 60,000 reads 0.9481 @ 40,164.) The dev picks of rule U (PL6): MuSiQue `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` 0.9550 at 15,400 (rung 2);
WebQSP `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` 0.9555 at 39,572 (rung 3); MetaQA `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,984 (rung 0). MultiHop-RAG, SQuAD, HotpotQA and 2Wiki: 50.0, 15.0, 77.3 and 908.2 dev nodes, all far inside 2,500.
Facts that the design rests on, each read before this page:
1. **The curve is smooth and gentle, not a cliff.** Every halving of the nodes costs ALL: MuSiQue 0.9640 at 15,400 to 0.9350 at 9,076 to 0.8945 at 2,980; WebQSP 0.9474 at 39,530 to 0.9328 at 10,817 to 0.8802 at 2,520; MetaQA 0.9582 at 4,973 to 0.9244 at 3,600 to 0.8999 at 2,699.
   **At 2,500 none of the three reaches the target on any cell read so far** (dev, inside 2,500: MuSiQue 0.8898, WebQSP 0.8759, MetaQA 0.8978).
2. **WebQSP did not reach 0.95 on eval at any size** (0.9474 at 39,530; 0.9481 at 40,164) although its dev pick read 0.9555: its U95 pick is a dev reach that eval does not confirm, and the call reads LEVEL against the bounded pick in the preview.
3. **The dev frontier has no step in two stretches**: MuSiQue between 3,020 and 5,783 nodes and WebQSP between 6,071 and 10,811, because PL6's ranked cells sit at V x f / 4 (2,500, 5,000, 7,500 at V = 10,000 and so on) and the earlier cells at V <= 5,000 stop at M = 1,600.
   A ceiling between those points has no cell cut for it; the ladder below fills them.
4. The earlier diagnoses stand (PL4 section 14.5 D2, PL5 section 1.2 D3, post-hoc, labels read; MuSiQue's from the coordinator's reading of PL6's ladder): WebQSP's shortfall at the bound is the per-seed and per-frontier caps, MetaQA's the hop-3 budget, MuSiQue's the cap-cut golds. Each is a reason for a wide walk; each wide walk is what the
   restriction below trims. The three families of walks PL6 used (c100 with a hop-3 or hop-2 shape, and the budget-following walks that have no caps) answer them one by one, and are kept unchanged.
5. **The links are the larger lever on the pool's size and are being tightened separately** (section 0, 25 Nov 23:00; the UL2 reports regenerated 23:05): on WebQSP the W2_S / R.mention label holds 500,445,467 of B0's 507,129,030 pairs (98.7%) at 0.009% precision
   and 16.8% gold-slot reach; MuSiQue's R.mention is 69.2% of B0 at 0.51% precision and 18.4% reach; MetaQA's 70.1% at 0.084% and 11.4%; K.knn is the best-precision label on most arms. A walk that spends 40,000 visited nodes on WebQSP spends them mostly on such pairs.
   This page does not change a link; it maps what the current links cost.

## 2. What is fixed (PL2's to PL6's pages, by reference)

Everything of PL2 sections 2 to 4, PL3 sections 2 and 4, PL4 sections 2, 4 and 6, PL5 section 4 and PL6 sections 2 and 4 (the walks, `cs`, `vr`, `sp`, `rk`, `fu`, `rr`, `eq`, `s10`, `s20`, the budget-following walk, the bases, the sizes, the measures, the statistics)
is unchanged and read by the same code (`zpl2g.py` to `zpl6g.py`, imported, not edited): the substrate, the dev carves and eval carves, the frozen first stage for selection and confirmation, the seeds, ALL = `all_gold_at_pool`, mean nodes = `candidates_mean`,
Wilson 95% on ALL, the paired bootstrap of 1,000 resamples, the floors 0.95 and 0.98, `Z3.status_of`. **PL2 to PL6's 6,078 to 6,546 cells per dataset (the table of `outputs/pl6/<d>/select.json`) are this page's rung 0** and are read from the filed tables; PL7 recomputes none of them for its reading
(it recomputes 312 of them as controls, section 8). The one thing this page adds to the engine is the visited budget V = 2,500 (the h2 and h3 c100 walks and the h2 and h3 budget-following walks), added to the settings table after the earlier hashes are cached.

## 3. What this round tests, and what it does not

| name | here |
|---|---|
| the restriction ladder (section 4) | ceilings K = 2,500 x 2^(j/2) rounded to 500: 2,500, 3,500, 5,000, 7,000, 10,000, 14,000, 20,000, 28,000, 40,000; a dataset is read at 2,500 and at every step below its PL6 pick's dev nodes |
| one rule on all seven (section 4) | R7(K): over the cells within K dev nodes, the smallest nodes with dev ALL >= 0.95, else the highest ALL |
| the hard cut (section 5) | a ranked cell `+top{K - b}` whose pool has at most K nodes by construction, for every K the walk is wider than |
| a new walk, order, window, key, rank (a parent-conditioned rank for the cut, a hop-weighted schedule, label-ordered caps) | **not in this round**; PL5 section 12 names them and each is declared on its own page |
| a link change (dropping or keeping a label or a key) | **not in this round**: declared separately (section 10) |

## 4. The ladder and the rule, exactly

**Ceilings.** The ladder is L = (2,500, 3,500, 5,000, 7,000, 10,000, 14,000, 20,000, 28,000, 40,000), 2,500 x 2^(j/2) rounded to the nearest 500: a step is a factor of about 1.41, two steps a halving. Per dataset the ceilings read are **2,500 and every step of L below the dev mean nodes of
PL6's U95 pick**: MuSiQue (pick 15,400) 2,500, 3,500, 5,000, 7,000, 10,000, 14,000; WebQSP (pick 39,572) 2,500 to 28,000 (eight steps); MetaQA (pick 4,984) 2,500 and 3,500. **MultiHop-RAG, SQuAD, HotpotQA and 2Wiki are read at 2,500 only**: their U95 picks (50.0, 15.0, 77.3, 908.2 nodes)
are inside it, so R7(2,500) is their PL6 pick by construction (the universality check of section 6), they have no PL7 cells and no host row.

**The rule R7(K)** (dev, frozen lists, per dataset and per ceiling K, over PL2 to PL6's table plus the PL7 cells of the dataset): among the cells whose dev mean nodes are at most K, **the smallest mean nodes with dev ALL >= 0.95** (ties to the higher ALL, then the name);
if no cell within K reaches 0.95, **the highest dev ALL** (ties to fewer nodes, then the name), and the dataset is SHORT at K. R98(K) likewise at 0.98, beside. `R95_old(K)` is the same rule over PL2 to PL6's cells alone: it says what the PL7 cells added at K.
R7(K) is PL6's rule U with a ceiling; at K = 2,500 on PL2 to PL5's cells it is PL5's bounded pick, which is the incumbent every call is read against.

## 5. The cells (fixed at this commit; nothing is added after a number is read)

For a dataset with a ladder (MuSiQue, WebQSP, MetaQA), with V in the visited budgets **V = 2,500 x 2^j up to the visited budget of PL6's pick's walk** (MuSiQue 2,500, 5,000, 10,000, 20,000; WebQSP those and 40,000; MetaQA 2,500 and 5,000):
**PL6's 13 walks** (`zpl6g.py` section 4: nine c100 walks and four budget-following walks; written for V by `walks7(V)`, equal to PL6's at V = 10,000, 20,000, 40,000) x the 3 bases (`rrf1000_b25`, `rrf1000_b400`, `rrf1000_b1600`, budgets b = 25, 400, 1,600):
- **the union cell** `base+walk` (at V = 2,500 new; at V >= 5,000 a cell that PL2 to PL6 already hold, here as a control, section 8);
- **the hard cut** `base+walk+top{K - b}` for every ceiling K of the dataset with K < V and K - b >= 1: the first K - b nodes of the walk's own cosine order (PL5 section 4: the ranked cells rank the walk's nodes by cosine to the question) joined to the base's top b, so **the pool has at most K nodes** (its maximum, not only its mean).

| dataset | visited budgets | walks | cells | held already (controls) | new | cell list sha256 (`P2.cells_sha256`) |
|---|---|---|---|---|---|---|
| musique | 2,500; 5,000; 10,000; 20,000 | 52 | 624 | 117 | 507 | `373ff76a51f4479b57b653b28606ffd8c451286dac091028d73ee659e949b19c` |
| webqsp | 2,500; 5,000; 10,000; 20,000; 40,000 | 65 | 975 | 156 | 819 | `ebe7442dd61ec5260f351da9b38c7fb6e4a228755ea9052127f9ea0e62fcfc7e` |
| metaqa | 2,500; 5,000 | 26 | 156 | 39 | 117 | `0e9ff2e94427d323aaa5810aa024889bf57e42fd9674f949fa6ae5a5d7d0f9c4` |

(`outputs/pl7/cells.json`; all 975 distinct names parse; the other four datasets have no cells.) **Why these cells, said so that nobody reads them as neutral.** The 13 walks are PL6's 13, which were themselves chosen from the walks of the highest-ALL cells above the bound in PL4's tables and from PL5's
budget-following walks; PL7 does not tune them. The visited budgets and the ceilings are one geometric ladder fixed in advance; they are not placed where a dev number is best. The bases and the cut's sizes are the ladder's own (K - b), so a pool never exceeds its ceiling. **What the cut cannot do:** it ranks the walk's nodes by cosine to the question
only, so a node reached by a low-cosine bridge is the first to go (PL5 section 12 names a parent-conditioned rank for that; it is not in this round).

## 6. Confirmation and the calls

**Confirmation** (eval, nothing re-chosen, frozen lists): for every dataset and ceiling the cells R95_all(K), R95_old(K) and R98_all(K) name, PL5's bounded pick (the incumbent), PL6's U95 and U98 picks and PL2's own construction. Cells PL5's or PL6's confirmation grids already hold are read from them, not recomputed
(`zpl7g.py confirm` builds only the others, in three parts per dataset).

**The call per dataset and ceiling** (eval, paired over the same questions; one generator per dataset, ceiling and comparison, `numpy.random.default_rng([20261125, d, K, c])`, d the dataset's place in the order MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MuSiQue, MetaQA, WebQSP; the status is PL3's `status_of`, clause by clause as in PL6 section 5):
- `vs_bounded`: R95_all(K) against PL5's bounded pick: the paired ALL difference, the nodes ratio, the status and the ALL-only status. **It decides only at K = 2,500.**
- `vs_u`: R95_all(K) against PL6's U95 pick: **what the restriction costs in ALL** (the paired difference with its interval) and the nodes ratio.
- `vs_without_pl7_cells`: R95_all(K) against R95_old(K): what the PL7 cells added at K.

**Filed beside every call (the price, deciding nothing, and the number the user weighs):** the eval ALL with its Wilson interval, the mean, 95th-percentile and maximum nodes per question, the nodes as a multiple of 2,500, the nodes as a share of the graph, the dev ALL and dev nodes of the pick, whether it reached 0.95 on dev and on eval.
**The knee** of a dataset is the smallest ceiling of its ladder whose R95_all pick reaches 0.95 on eval (point estimate); the page states its eval nodes as a multiple of 2,500 and what the step below it costs in ALL. A dataset with no knee on its ladder says so.
**The floor caveat (my definition; the user or the coordinator may correct it).** A pick that reaches 0.95 on eval by its point estimate whose Wilson lower end is below 0.95 is flagged `at_floor`: its ABOVE or its reach rests on a point that lies within sampling error of the floor
(about 1 point at 2,000 questions), so it is read as "at the floor", not as a margin over it; the page prints the flag beside every status and the interval beside the point.

**CARRY_PL7 is read at K = 2,500 only**: all seven read, **no dataset BELOW** the bounded pick at K = 2,500, at least one ABOVE, and the universality check (on every dataset whose bounded pick reached 0.95 on dev, R95_all(2,500) is that cell: it holds by construction; it is filed as a check of the code, not a finding).
INCOMPLETE while fewer than seven are read; otherwise NOT_CARRIED (PL5's bounded rule stays what PL2 to PL5 used). **No reading above 2,500 is carried, whatever it reads.** The page says per dataset how far above 2,500 the knee stands and what each step down costs in ALL.

## 7. Stop rule

The ladder L, the visited budgets, the 13 walks, the bases, the cut's size K - b, rule R7(K), the readings, the calls, their statistics, the carves, the knee and the floor flag, and the datasets' ceilings (which depend on PL6's dev picks' mean nodes, stored as constants in `zpl7g.py` `READ6`) are fixed by this commit; every output records the sha256 of its cell list and the code-path hash,
`zpl7g.py` refuses a grid whose cell list is not the declared one, a select whose heads carry another code-path hash and a call over selects that differ (G-U). Nothing is added after a number is read. A dataset that stays SHORT at 2,500 is reported with D1 and D2 at its R95_all(2,500) cell (`zpl5r.py miss`, post-hoc, labels read, deciding nothing),
and the next change is declared on its own page.

## 8. Gates (each is a failure that stops that dataset's reading; none of its numbers is read)

- **G-P (the pinned code).** The code-path hashes of PL2 to PL6 computed by this process before V = 2,500 is added equal the ones filed at PL5's and PL6's self-tests (PASSED before this commit).
- **Engine self-test (PASSED before this commit, `outputs/pl7/selftest.json`).** Cell lists and sha256s, every name parses, no repeated name, every cut's M + b is a ceiling of its dataset, PL7's walk list at V = 10,000, 20,000, 40,000 equals PL6's; the engine's 13 walks at V = 2,500 against PL5's per-node reference on 1 real question each
  of MetaQA and MuSiQue: 13 checks each, 0 mismatches; 30 hard-cut cells on each: no pool larger than its ceiling.
- **G-C (the controls).** Every PL7 cell whose name PL2 to PL6 already filed (MuSiQue 117, WebQSP 156, MetaQA 39: the union cells at V >= 5,000) equals the filed array **element for element on every dev question** (present counts and pool sizes). A difference stops the dataset's select and files `gc_FAIL.json`.
  It is a stronger gate than PL6's G-R (the whole carve, not 100 questions) and costs nothing: the cells are computed anyway. The select (a table read) runs on the laptop, which holds the earlier grids.
- **G-F, G-D.** No pool function is new; the earlier rounds' G-F (gold-free) and G-D (determinism) hold for the code this page imports, which G-P pins.
- **G-U (universal).** One code-path sha256 in every grid head; `select`, `confirm` and `call` refuse a head that differs.

## 9. What the round can and cannot show

It can show the ALL-for-nodes curve of the current links at ceilings a factor of about 1.41 apart down to 2,500, with pools that never exceed their ceiling; whether a wide walk with a hard cut keeps more golds per node than PL5's narrow bounded pick at 2,500 (the question PL6's table could not answer, because it has no
cells there); how far above 2,500 the knee of each dataset stands; and what each step down costs in ALL, with the node cost beside it. **It is likely to end SHORT at 2,500 on MuSiQue, WebQSP and MetaQA** (fact 1 above: no cell read so far reaches 0.95 inside 2,500, and the dev frontier at 3,000 is held by the same V = 2,000 walks as the bounded pick); if so
the answer is the price curve, not a carry, and the lever is on the link side (section 10). It cannot reach a gold that no walk reaches from the first stage (PL4 14.5), cannot reorder the cut (cosine only), and says nothing about links after tightening.

## 10. Provisional pools: what carries to the tightened links, and what is redone

The user's sequencing (section 0) puts the link set before the vectors and models, and a pool re-run on the tightened links after it. **This page's pools are provisional until the link set is final.**
- **Carries to the tightened links unchanged:** the target (ALL >= 0.95 at the fewest nodes), the ladder L and the rule R7(K), the calls and their statistics, the knee, the floor flag, the gates, the cell recipe (13 walks x bases x visited budgets x hard cuts). A re-run is `zpl7g.py` pointed at the new stores: no pool function changes (the walks read the typed, ner and knn stores as stores; PL2 to PL7 never read a label).
- **Re-run on the tightened links, nothing reused:** every grid (cells, dev arrays, picks, knees, p95, the PL6 ladder's rungs), because a link set changes which nodes a walk reaches and what a visited budget buys. **The constants that depend on the current links** are `READ6` (PL6's dev picks that set each dataset's ceilings and visited budgets) and the visited budgets
  2,500 to 40,000 themselves: on tightened links the knee may move far below WebQSP's 40,000 (fact 5), so the re-run's ladder starts from the tightened links' own unbounded pick, by the same rule (the PL6 ladder, then this ladder), not from these numbers.
- **One re-run, not several:** the host cost of PL2 to PL7 on three datasets is many host CPU-hours; the re-run should be the rules of this page (PL6's ladder to find the unbounded pick and this page's ladder to walk it down) on the final link set only, and nothing between now and then should add host rows beyond this page's grids and confirmations.
- **Why a re-run on tightened links is sound, and one condition on it:** the pools are functions of the stores and the question and read no label, so a tightened store changes only what they reach. The condition is on the link selection: keeping or dropping a label or a key (a per-label rule) against gold reach on the **dev carves only** and
  confirming on eval keeps eval clean; a per-pair keep-or-drop against gold links would fit the pairs the eval questions use and must not be used. That is the link stage's rule to declare; it is named here because it decides whether PL7's eval reading stays an out-of-sample reading on the new links.
  I see no reason the pool rounds must not be redone once on tightened links; the reason to keep this page's rows small is that they are redone.

## 11. Code, rows, outputs

- **New module:** `outputs/mp_unified/zpl7g.py` (commands `cells`, `selftest`, `grid`, `select`, `confirm`, `call`, `rows`; `selftest`, `grid`, `select` and `confirm` take `--host`), committed with this page. It imports `zpl6g.py` and edits none of the earlier modules.
- **Host rows** (`outputs/pl7/feed_rows.txt`, nine rows, none depends on another; the dev grid in three parts per dataset, 2 cpus, MuSiQue and MetaQA 3 GB, WebQSP 10 GB as PL6's): `pl7-musiqueg0` to `g2`, `pl7-webqspg0` to `g2`, `pl7-metaqag0` to `g2`. Their inputs are the code and `outputs/pl7/selftest.json`. **Expected host cost** (not measured for PL7; from PL6's measured host seconds per part
  of about 500 dev questions: MuSiQue V = 10,000 about 2,700 s and V = 20,000 about 4,300 s, WebQSP V = 10,000 about 1,650 s, V = 20,000 about 2,200 s and V = 40,000 about 2,500 to 3,200 s, with an allowance for V = 2,500 and 5,000 and for the cut cells): MuSiQue about 9,500 s per part (2.6 h), WebQSP about 9,000 s per part (2.5 h), MetaQA under 1 h; good to a factor of 1.5.
- **Laptop** (after the grids are fetched): `select --ds D` on all seven (G-C first; seconds to minutes), then the three confirmation parts per ladder dataset as host rows (`zpl7g.py rows --confirm`, inputs `outputs/pl7/<d>/select.json`), then `call`, then `zpl5r.py miss` at the 2,500 picks of the SHORT datasets and the results section appended to this page.
- **Outputs:** `outputs/pl7/{cells.json, selftest.json, feed_rows.txt, call.json}`, `outputs/pl7/<d>/{grid_dev_frozen.part*, select.json, grid_eval_frozen_confirm.part*, gc_FAIL.json if any}`; the arrays (`*.npz`) are not committed (as PL2 to PL6).

## 12. What follows

1. PL5's and PL6's own results pages are written first (SQuAD's PL5 confirmation is the last read); this page's rows then run on the host. The picks, the calls, the knees and the price tables are filed in section 13 of this page.
2. If CARRY_PL7 (all seven at 2,500, no BELOW, one ABOVE) the user decides whether the pools downstream take the 2,500 picks; if NOT_CARRIED, PL5's bounded rule stays and the price curve is the finding.
3. The link stage's tightening (section 10) comes before any further pool round; the ladders of PL6 and PL7 are then re-run once on the final link set.
4. Next changes, names only: a parent-conditioned rank for the cut, a hop-weighted schedule, label-ordered caps (PL5 section 12).

## 13. Disclosed

- **The ladder's ceilings and visited budgets were set after I had read PL6's dev and eval pick tables** (section 1): the three datasets and their upper steps come from PL6's U95 picks (`READ6`). I did not tune any step to a dev number; the ladder is one geometric sequence, and a finer or coarser one changes the resolution of the knee, not the rule.
- **The 13 walks are PL6's 13**, partly chosen from PL4's above-bound tables that PL6 had seen (PL6 section 4); the V = 2,500 and the hard-cut cells are new and are not chosen from any number.
- **About 1,750 new cells are added to about 6,400 per dataset** in the selection: a winner's dev ALL is optimistic, so every call is a paired interval on eval; the dev numbers of the table are the selection's, not a result.
- **The floor caveat is my definition** (section 6); the coordinator relayed the wish, not the wording.
- **The hard cut ranks by cosine only**; a different rank for the cut is another page.
- **The host-time estimates are from PL6's seconds**, not measured for PL7 (section 11).
- **The four datasets inside 2,500 have no PL7 cells and no host row**: their K = 2,500 pick is PL6's by construction, and their eval record is PL5's confirmation of it.
- **PL6's results page is not yet written and its call is preliminary (6 of 7 read)**; PL7 reads PL6's filed select files, which do not change when PL6's page is written. If SQuAD's PL5 confirmation changes any select (it cannot: selects are dev), this page is not edited; a difference would show in the record.
- **The pools are on the current links and provisional** (section 10): the user's sequencing puts link tightening before any further pool use, so every number here is a map of the current links, not a carry onto the final link set.
- The six datasets' graph is the old package graph (as PL2 to PL6 disclosed); MultiHop-RAG reads its B0 layout. The UL2 layout is not materialized on the six.
