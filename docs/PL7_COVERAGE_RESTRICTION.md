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

## 14. Results (26 Nov 2026; 7 of 7 datasets read)

Everything here is read by the rules of sections 4 to 8 from the filed records (`outputs/pl7/<d>/{select,miss}.json`, `outputs/pl7/call.json`, the grid heads, `outputs/pl7/selftest.json`; PL5's `outputs/pl5/<d>/miss.json` for the equality check of 14.6) by `outputs/mp_unified/zpl7g.py`
(`select`, `confirm`, `call`) and `outputs/mp_unified/zpl5r.py miss --pl7`. No PL5 or PL6 record is changed. Nothing was changed in the rule, the cells or the engine after a number was read. The declaration's section 12 (item 1) says the results are filed in section 13; section 13 of this page is the
declaration's own disclosures, so the results are section 14.

### 14.1 Gates and cost

The engine `zpl7g.py` and its code-path hash `58cea08c7a12` are the declaration commit's (24f4892e): the same hash is in the self-test, in all 7 select files and in every grid head (`code_path_equal_everywhere` true; G-U PASS). The cell-list sha256 of every dev grid head equals the one the declaration states (section 5): MuSiQue `373ff76a51f4`, WebQSP `ebe7442dd61e`, MetaQA `0e9ff2e94427`. The self-test (`selftest.json`) is the declaration's own: PASS in 56.6 s.

**G-C (the controls, element for element on every dev question)** PASSED on all three: MuSiQue 117 control cells x 1,534 questions (358,956 elements), 0 differ; WebQSP 156 control cells x 1,507 questions (470,184 elements), 0 differ; MetaQA 39 control cells x 1,497 questions (116,766 elements), 0 differ.

Cost (host rows, three parts per dataset; the grids are `outputs/pl7/<d>/grid_dev_frozen.part*of3` and `grid_eval_frozen_confirm.part*of3`; `select` and `call` ran on the laptop in seconds):

| dataset | dev grid: cells | seconds per part (sum) | peak GB per part | declared estimate (section 11) | eval confirmation: cells built | seconds per part (sum) | peak GB |
|---|---|---|---|---|---|---|---|
| MuSiQue | 624 | 7,669, 8,070, 8,162 (23,901) | 0.62 | about 9,500 s per part | 2 | 366, 366, 418 (1,151) | 0.64 |
| WebQSP | 975 | 4,824, 4,429, 4,444 (13,697) | 7.75 | about 9,000 s per part | 1 | 528, 526, 461 (1,515) | 6.51 |
| MetaQA | 156 | 404, 391, 428 (1,224) | 0.38 | under 1 h | 1 | 14, 13, 14 (41) | 0.32 |

### 14.2 The call at K = 2,500 (read 7 of 7: NOT_CARRIED)

R7(2,500) = R95_all(2,500) of the reading `all` over PL2 to PL6's cells and the PL7 cells, against PL5's bounded pick (what PL2 to PL5 carried); eval, frozen lists, paired bootstrap, one generator per dataset, ceiling and comparison (section 6). `at floor` is the floor caveat of section 6 (my definition): a pick that reaches 0.95 on eval
by its point estimate while its Wilson lower end is below 0.95.

| dataset | R7(2,500), the pick | the same cell as PL5's bounded pick | eval ALL [Wilson 95%] | mean / p95 / max nodes per question | x 2,500 | share of graph | reaches 0.95 on eval | at floor | ALL difference vs the bounded pick (points) [95% CI] | nodes ratio | status | R98(2,500): eval ALL at mean nodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` | yes | 0.9690 [0.9572, 0.9776] | 50.0 / 50 / 50 | 0.02 | n/a | yes | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 |
| SQuAD | `rrf1000_b15` | yes | 0.9554 [0.9515, 0.9589] | 15.0 / 15 / 15 | 0.01 | n/a | yes | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | yes | 0.9634 [0.9589, 0.9674] | 77.9 / 125 / 125 | 0.03 | n/a | yes | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | yes | 0.9609 [0.9573, 0.9641] | 917.0 / 1,192 / 1,200 | 0.37 | n/a | yes | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9656 at 2,211.2 |
| MuSiQue | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | yes | 0.8846 [0.8712, 0.8967] | 2,438.2 / 2,633 / 2,746 | 0.98 | 2.07% | no | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8846 at 2,438.2 |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | yes | 0.8968 [0.8906, 0.9027] | 2,335.9 / 2,376 / 2,389 | 0.93 | 5.40% | no | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 |
| WebQSP | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | yes | 0.8789 [0.8614, 0.8944] | 2,094.6 / 2,173 / 2,189 | 0.84 | 0.08% | no | no | +0.00 [+0.00, +0.00] | 1.000 | LEVEL | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8789 at 2,094.6 |

- **The call: NOT_CARRIED** (BELOW 0, ABOVE 0, LEVEL 7; read 7 of 7). CARRY_PL7 needs all seven read, no BELOW against the bounded pick at 2,500, at least one ABOVE, and the universality check; with no ABOVE the call is NOT_CARRIED and PL5's bounded rule stays what PL2 to PL5 used (section 6).
- **R7(2,500) is PL5's bounded pick on all seven**: the same cell, the ALL difference exactly 0 and the nodes ratio exactly 1, so no dataset can read ABOVE or BELOW. The PL7 cells (507 on MuSiQue, 819 on WebQSP, 117 on MetaQA) are not the dev pick at 2,500 on any of them (14.5).
- **The universality check** (on every dataset whose bounded pick reached 0.95 on dev, R7(2,500) is that cell): MultiHop-RAG, SQuAD, HotpotQA, 2Wiki; unchanged on all of them: yes. It holds by construction (section 6) and is a check of the code, not a finding.
- **Eval, frozen lists: four datasets reach 0.95 inside 2,500** (MultiHop-RAG 0.9690, SQuAD 0.9554, HotpotQA 0.9634, 2Wiki 0.9609), at 0.02, 0.01, 0.03
  and 0.37 times 2,500 nodes; none is flagged at the floor. **Three do not**: MuSiQue 0.8846, MetaQA 0.8968, WebQSP 0.8789, short by 6.54, 5.32 and 7.11 points; their Wilson upper ends (0.8967, 0.9027, 0.8944) are below 0.95.
  The same numbers as PL5's section 14.3, as they must be.

### 14.3 The restriction ladder: what each step down costs, per dataset

Eval, frozen lists; the cell of each ceiling was chosen on dev (rule R7(K)) and read once on eval. The three comparisons are paired over the same questions. **Read the status column as the rule gives it, and the ALL-only status beside it**: the `vs PL6's U95 pick` status reads ABOVE where the pick uses fewer nodes (clause ABOVE 2:
nodes-ratio upper end below 1) even when ALL is lower; the ALL difference beside it says which. Nothing above 2,500 is carried, whatever it reads (section 6).

**MuSiQue** (117,534 nodes in the graph; 2,417 eval questions)

| K | R95_all(K), the pick | dev ALL at dev nodes | eval ALL [Wilson 95%] at mean nodes (p95 / max) | x 2,500 | share of graph | reaches 0.95 on eval | step from the row above: eval ALL (points), nodes ratio | vs PL5's bounded pick: ALL diff (points) [95% CI], nodes ratio, status | vs PL6's U95 pick: ALL diff (points) [95% CI], nodes ratio, status | vs R95_old(K) (what the PL7 cells added) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,500 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 0.8898 at 2,464.3 | 0.8846 [0.8712, 0.8967] at 2,438.2 (2,633 / 2,746) | 0.98 | 2.07% | no | - | +0.00 [+0.00, +0.00], 1.000, LEVEL (ALL-only LEVEL) | -7.94 [-8.98, -6.79], 0.158, BELOW (ALL-only BELOW) | same cell |
| 3,500 | `rrf1000_b1600+FULL:h2_c100_v2500~s10-cs-vr` | 0.9074 at 3,489.4 | 0.8999 [0.8873, 0.9112] at 3,449.1 (3,770 / 3,982) | 1.38 | 2.93% | no | +1.53, 1.41 | +1.53 [+0.79, +2.23], 1.415, LEVEL (ALL-only ABOVE) | -6.41 [-7.41, -5.42], 0.224, BELOW (ALL-only BELOW) | `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` 0.8945 at 2,980.2: +0.54 [-0.08, +1.12], ratio 1.157, LEVEL (ALL-only LEVEL) |
| 5,000 | `rrf1000_b1600+FULL:h2_c100_v2500~cs-vr` | 0.9081 at 3,518.3 | 0.8990 [0.8864, 0.9104] at 3,480.7 (3,803 / 4,004) | 1.39 | 2.96% | no | -0.08, 1.01 | +1.45 [+0.87, +2.11], 1.428, LEVEL (ALL-only ABOVE) | -6.50 [-7.45, -5.50], 0.226, BELOW (ALL-only BELOW) | `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` 0.8945 at 2,980.2: +0.46 [+0.00, +0.91], ratio 1.168, LEVEL (ALL-only LEVEL) |
| 7,000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9355 at 5,783.4 | 0.9222 [0.9109, 0.9322] at 5,742.4 (6,067 / 6,354) | 2.30 | 4.89% | no | +2.32, 1.65 | +3.76 [+2.98, +4.59], 2.355, LEVEL (ALL-only ABOVE) | -4.18 [-5.05, -3.27], 0.373, BELOW (ALL-only BELOW) | same cell |
| 10,000 | `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu+top7500` | 0.9420 at 9,086.3 | 0.9350 [0.9245, 0.9442] at 9,076.2 (9,100 / 9,100) | 3.63 | 7.72% | no | +1.28, 1.58 | +5.05 [+4.18, +6.04], 3.722, LEVEL (ALL-only ABOVE) | -2.90 [-3.64, -2.19], 0.589, BELOW (ALL-only BELOW) | same cell |
| 14,000 | `rrf1000_b1600+FULL:h2_c100_v20000~s20-cs-rk+top15000` | 0.9492 at 13,970.0 | 0.9429 [0.9329, 0.9515] at 13,614.8 (16,600 / 16,600) | 5.45 | 11.58% | no | +0.79, 1.50 | +5.83 [+4.88, +6.74], 5.584, LEVEL (ALL-only ABOVE) | -2.11 [-2.94, -1.28], 0.884, BELOW (ALL-only BELOW) | same cell |
| PL6's U95 pick | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | - | 0.9640 at 15,400.0 (dev nodes 15,400.0) | 6.16 | - | yes | - | - | - | - |

**WebQSP** (2,592,894 nodes in the graph; 1,503 eval questions)

| K | R95_all(K), the pick | dev ALL at dev nodes | eval ALL [Wilson 95%] at mean nodes (p95 / max) | x 2,500 | share of graph | reaches 0.95 on eval | step from the row above: eval ALL (points), nodes ratio | vs PL5's bounded pick: ALL diff (points) [95% CI], nodes ratio, status | vs PL6's U95 pick: ALL diff (points) [95% CI], nodes ratio, status | vs R95_old(K) (what the PL7 cells added) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,500 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 0.8759 at 2,094.2 | 0.8789 [0.8614, 0.8944] at 2,094.6 (2,173 / 2,189) | 0.84 | 0.08% | no | - | +0.00 [+0.00, +0.00], 1.000, LEVEL (ALL-only LEVEL) | -6.85 [-8.25, -5.72], 0.053, ABOVE (ALL-only BELOW) | same cell |
| 3,500 | `rrf1000_b400+STRUCT:h3_bf_v2500~cs-fu` | 0.8892 at 2,716.1 | 0.8902 [0.8734, 0.9050] at 2,718.4 (2,848 / 2,884) | 1.09 | 0.10% | no | +1.13, 1.30 | +1.13 [+0.60, +1.66], 1.298, LEVEL (ALL-only ABOVE) | -5.72 [-6.99, -4.66], 0.069, ABOVE (ALL-only BELOW) | `rrf1000_b1600+STRUCT:h2_bf_v2000~cs-fu` 0.8736 at 3,247.0: +1.66 [+0.53, +2.66], ratio 0.837, ABOVE (ALL-only ABOVE) |
| 5,000 | `rrf1000_b800+FULL:h2_bf_v5000~cs-fu` | 0.9018 at 4,994.7 | 0.8975 [0.8812, 0.9119] at 4,976.6 (5,700 / 5,763) | 1.99 | 0.19% | no | +0.73, 1.83 | +1.86 [+0.67, +2.99], 2.376, LEVEL (ALL-only ABOVE) | -4.99 [-6.12, -3.86], 0.126, ABOVE (ALL-only BELOW) | same cell |
| 7,000 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9257 at 6,070.5 | 0.9175 [0.9025, 0.9304] at 6,079.5 (6,455 / 6,556) | 2.43 | 0.23% | no | +2.00, 1.22 | +3.86 [+2.93, +4.86], 2.902, LEVEL (ALL-only ABOVE) | -2.99 [-3.86, -2.13], 0.154, ABOVE (ALL-only BELOW) | same cell |
| 10,000 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9257 at 6,070.5 | 0.9175 [0.9025, 0.9304] at 6,079.5 (6,455 / 6,556) | 2.43 | 0.23% | no | +0.00, 1.00 | +3.86 [+2.86, +4.79], 2.902, LEVEL (ALL-only ABOVE) | -2.99 [-3.93, -2.13], 0.154, ABOVE (ALL-only BELOW) | same cell |
| 14,000 | `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu` | 0.9416 at 10,810.7 | 0.9328 [0.9190, 0.9444] at 10,817.4 (11,270 / 11,514) | 4.33 | 0.42% | no | +1.53, 1.78 | +5.39 [+4.26, +6.59], 5.164, LEVEL (ALL-only ABOVE) | -1.46 [-2.26, -0.80], 0.274, ABOVE (ALL-only BELOW) | same cell |
| 20,000 | `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu` | 0.9416 at 10,810.7 | 0.9328 [0.9190, 0.9444] at 10,817.4 (11,270 / 11,514) | 4.33 | 0.42% | no | +0.00, 1.00 | +5.39 [+4.26, +6.59], 5.164, LEVEL (ALL-only ABOVE) | -1.46 [-2.20, -0.73], 0.274, ABOVE (ALL-only BELOW) | same cell |
| 28,000 | `rrf1000_b1600+FULL:h3_bf_v20000~cs-fu` | 0.9496 at 20,744.3 | 0.9421 [0.9291, 0.9528] at 20,737.5 (21,242 / 21,514) | 8.30 | 0.80% | no | +0.93, 1.92 | +6.32 [+5.19, +7.45], 9.901, LEVEL (ALL-only ABOVE) | -0.53 [-1.06, +0.00], 0.525, ABOVE (ALL-only LEVEL) | same cell |
| PL6's U95 pick | `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` | - | 0.9474 at 39,529.5 (dev nodes 39,571.6) | 15.81 | - | no | - | - | - | - |

**MetaQA** (43,234 nodes in the graph; 9,785 eval questions)

| K | R95_all(K), the pick | dev ALL at dev nodes | eval ALL [Wilson 95%] at mean nodes (p95 / max) | x 2,500 | share of graph | reaches 0.95 on eval | step from the row above: eval ALL (points), nodes ratio | vs PL5's bounded pick: ALL diff (points) [95% CI], nodes ratio, status | vs PL6's U95 pick: ALL diff (points) [95% CI], nodes ratio, status | vs R95_old(K) (what the PL7 cells added) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,500 | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8978 at 2,339.7 | 0.8968 [0.8906, 0.9027] at 2,335.9 (2,376 / 2,389) | 0.93 | 5.40% | no | - | +0.00 [+0.00, +0.00], 1.000, LEVEL (ALL-only LEVEL) | -6.14 [-6.64, -5.68], 0.470, BELOW (ALL-only BELOW) | same cell |
| 3,500 | `equal_rrf_budget_50+STRUCT:h3_c25_v4000` | 0.9259 at 3,477.2 | 0.9240 [0.9185, 0.9291] at 3,475.3 (4,038 / 4,044) | 1.39 | 8.04% | no | +2.72, 1.49 | +2.72 [+2.39, +3.07], 1.488, LEVEL (ALL-only ABOVE) | -3.42 [-3.96, -2.89], 0.699, BELOW (ALL-only BELOW) | same cell |
| PL6's U95 pick | `rrf1000_b25+STRUCT:h3_c100_v5000` | - | 0.9582 at 4,972.8 (dev nodes 4,983.6) | 1.99 | - | yes | - | - | - | - |

The other four datasets are read at 2,500 only (14.2): their U95 picks sit at 0.01 to 0.37 times 2,500.

### 14.4 The knee of each dataset

The smallest ceiling of the dataset's ladder whose R95_all pick reaches 0.95 on eval (point estimate), its eval nodes as a multiple of 2,500, and what the step below costs in ALL (section 6).

| dataset | knee ceiling | eval nodes at the knee | x 2,500 | what the step below costs in ALL | what the ladder says |
|---|---|---|---|---|---|
| MultiHop-RAG | 2,500 | 50.0 | 0.020 | none: the ladder starts at 2,500 | the dataset reaches 0.95 on eval inside 2,500 (`rrf1000_b25+FULL:h1_c100+top25` 0.9690); R7(2,500) is PL6's pick and PL5's bounded pick |
| SQuAD | 2,500 | 15.0 | 0.006 | none: the ladder starts at 2,500 | the dataset reaches 0.95 on eval inside 2,500 (`rrf1000_b15` 0.9554); R7(2,500) is PL6's pick and PL5's bounded pick |
| HotpotQA | 2,500 | 77.9 | 0.031 | none: the ladder starts at 2,500 | the dataset reaches 0.95 on eval inside 2,500 (`rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634); R7(2,500) is PL6's pick and PL5's bounded pick |
| 2Wiki | 2,500 | 917.0 | 0.367 | none: the ladder starts at 2,500 | the dataset reaches 0.95 on eval inside 2,500 (`rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609); R7(2,500) is PL6's pick and PL5's bounded pick |
| MuSiQue | none on the ladder | - | - | each step is in the ladder table: ending at 2,500 costs -7.94 [-8.98, -6.79] points of ALL against PL6's U95 pick (nodes ratio 0.158) | the last ceiling read (14,000) reads 0.9429 at 13,614.8 nodes; PL6's U95 pick 0.9640 at 15,400.0 (6.16 x 2,500) reaches 0.95 on eval |
| MetaQA | none on the ladder | - | - | each step is in the ladder table: ending at 2,500 costs -6.14 [-6.64, -5.68] points of ALL against PL6's U95 pick (nodes ratio 0.470) | the last ceiling read (3,500) reads 0.9240 at 3,475.3 nodes; PL6's U95 pick 0.9582 at 4,972.8 (1.99 x 2,500) reaches 0.95 on eval |
| WebQSP | none on the ladder | - | - | each step is in the ladder table: ending at 2,500 costs -6.85 [-8.25, -5.72] points of ALL against PL6's U95 pick (nodes ratio 0.053) | the last ceiling read (28,000) reads 0.9421 at 20,737.5 nodes; PL6's U95 pick 0.9474 at 39,529.5 (15.81 x 2,500) does not reach 0.95 on eval either |

**Three datasets have no knee on their ladders** (MuSiQue to 14,000, WebQSP to 28,000, MetaQA to 3,500). PL6's U95 pick is the known reach: MuSiQue 0.9640 at 15,400.0 nodes (6.16 times 2,500), MetaQA 0.9582 at 4,972.8 (1.99 times); WebQSP does not reach
0.95 on eval at any size read (0.9474 at 39,529.5 nodes, 15.81 times 2,500). Between the last ceiling and PL6's pick the ladder reads nothing (the next step of L lies above PL6's dev nodes, section 4), so where the knee lies inside that stretch is not read here.

### 14.5 What the PL7 cells added (the question of section 9)

Section 9 asked whether a wide walk with a hard cut keeps more golds per node than PL5's narrow bounded pick at 2,500. Dev, frozen lists: the best new cell inside 2,500 dev nodes against PL5's bounded pick, and the ceilings at which R95_all differs from R95_old (eval, paired):

| dataset | new cells | new cells inside 2,500 dev nodes | best new cell inside 2,500 (dev ALL at dev nodes) | PL5's bounded pick (dev) | difference (points, dev) | ceilings at which the PL7 cells changed the pick (eval, against R95_old(K)) |
|---|---|---|---|---|---|---|
| MuSiQue | 507 | 119 | `rrf1000_b400+FULL:h3_c100_v5000~cs-sp+top2100` 0.8514 at 2,500.0 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8898 at 2,464.3 | -3.85 | K = 3,500: +0.54 [-0.08, +1.12], nodes ratio 1.157, LEVEL (ALL-only LEVEL); K = 5,000: +0.46 [+0.00, +0.91], nodes ratio 1.168, LEVEL (ALL-only LEVEL) |
| WebQSP | 819 | 208 | `rrf1000_b25+FULL:h2_bf_v2500~s10-cs-vr` 0.8560 at 2,439.4 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8759 at 2,094.2 | -1.99 | K = 3,500: +1.66 [+0.53, +2.66], nodes ratio 0.837, ABOVE (ALL-only ABOVE) |
| MetaQA | 117 | 50 | `rrf1000_b400+FULL:h2_c100_v5000~cs-vr+top3100` 0.7422 at 2,065.4 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | -15.56 | none |

**At 2,500 the answer is no, on all three: the best new cell inside the ceiling reads 0.8514 on MuSiQue, 0.8560 on WebQSP and 0.7422 on MetaQA against 0.8898, 0.8759 and 0.8978** (-3.8, -2.0 and -15.6 points on dev).
Above 2,500 the cells added something at three of the thirteen ceilings read above 2,500 (sixteen on the three ladders, three of them at 2,500): MuSiQue at 3,500 (+0.54 points [-0.08, +1.12], nodes ratio 1.157, LEVEL) and 5,000 (+0.46 [+0.00, +0.91], 1.168, LEVEL), and WebQSP at 3,500
(`rrf1000_b400+STRUCT:h3_bf_v2500~cs-fu`, a budget-following walk at V = 2,500: 0.8902 at 2,718.4 nodes against `rrf1000_b1600+STRUCT:h2_bf_v2000~cs-fu` 0.8736 at 3,247.0; +1.66 points [+0.53, +2.66], nodes ratio 0.837, ABOVE). At no other ceiling did a PL7 cell win on dev. **No hard-cut cell (`+top{K - b}`) is the pick at any ceiling on any dataset**: the three PL7 cells that won are union cells of the V = 2,500 walks (`base+walk` at V = 2,500), whose pools are not cut to the ceiling (their maxima are 3,982, 4,004 and 2,884 nodes on eval).

### 14.6 Still short at 2,500: D1 and D2 (post-hoc, labels read; 300 dev questions evenly spaced over the carve; `zpl5r.py miss --pl7`)

The diagnostic of PL5's section 14.6 at R95_all(2,500) of the three SHORT datasets, which is PL5's chosen cell on each (14.2). **D1** (each gold missing from the pool: R ranked past M, C cut by the caps, the window or the order but inside the uncapped three-hop reach of its seeds, U beyond it):

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (mean, dev) | equals PL5's `miss.json` (cell, totals, stages) |
|---|---|---|---|---|---|---|---|---|
| MuSiQue | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 36 of 300 (0.120) | 39 (1.08 per such question) | 0 | 39 | 0 | 2,462 | yes |
| WebQSP | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 31 of 300 (0.103) | 81 (2.61 per such question) | 0 | 81 | 0 | 2,094 | yes |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | 33 of 300 (0.110) | 375 (11.36 per such question) | 0 | 373 | 2 | 2,337 | yes |

**D2** (which stage cut each C gold, from a traced walk of the cell):

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|
| MuSiQue | 39 | 0 | 18 | 0 | 9 | 0 | 12 | yes |
| WebQSP | 81 | 2 | 35 | 5 | 19 | 16 | 4 | yes |
| MetaQA | 373 | 0 | 15 | 12 | 0 | 264 | 82 | yes |

**Read:** the records equal PL5's exactly (same cell, same 300 questions, a deterministic walk), so PL5's reading of them stands unchanged: the golds the 2,500-node picks miss are cut by a per-seed or per-frontier cap or by the visited budget, not ranked out and (but for two on MetaQA) not unreachable. What this round adds is on dev ALL (14.5): a wider walk cut to 2,500 nodes did not keep more golds than the narrow bounded walk. **How the ceiling is cut is a separate question from which golds the cut keeps**: the cut ranks the walk's nodes by cosine to the question only (section 5), and a gold that arrives through a low-cosine bridge is the first to go; I did not test that on this page.

### 14.7 What the call means, and what it leaves unread

- **NOT_CARRIED, 7 of 7 read, is a statement about the 2,500 pick, not about the curve.** With one rule on all seven, the node count restricted back to 2,500 reads PL5's bounded pick on every dataset; nothing in the PL7 grid improves on it there. The four datasets that reach 0.95 reach it at 0.01 to 0.37 times 2,500;
  the other three are SHORT at 2,500 and stay so.
- **The price curve is the finding** (section 9 said it would be), eval, frozen lists: MuSiQue 0.8846 at 2,438 nodes, 0.8999 at 3,449 (1.38 times 2,500), 0.9222 at 5,742 (2.30), 0.9350 at 9,076 (3.63), 0.9429 at 13,615 (5.45),
  PL6's 0.9640 at 15,400 (6.16); WebQSP 0.8789 at 2,095, 0.8902 at 2,718 (1.09), 0.8975 at 4,977 (1.99), 0.9175 at 6,080 (2.43), 0.9328 at 10,817 (4.33), 0.9421 at 20,738 (8.30),
  PL6's 0.9474 at 39,530 (15.81), never 0.95; MetaQA 0.8968 at 2,336, 0.9240 at 3,475 (1.39), PL6's 0.9582 at 4,973 (1.99).
  **To the user's weighing, on the current links: PL6's pick reaches 0.95 on MetaQA at 1.99 times 2,500 nodes (the ladder's 1.39 times does not) and on MuSiQue at 6.16 times (the ladder's 5.45 times does not); on WebQSP no size read reaches it.** The curve has no cliff: the largest rise from one ceiling's pick to the next is +2.72 points of eval ALL, for about 1.5 to 1.7 times the nodes (section 1 fact 1).
- **Every step above 2,500 buys ALL for nodes, and the rule's status never calls it ABOVE**: at every ceiling above 2,500 on the three ladders the status against the bounded pick is LEVEL while the ALL-only status is ABOVE (more ALL for more nodes: the price clause does not fire). A reader who wants the ALL gain at 1.09 times 2,500 on WebQSP (+1.13 points [+0.60, +1.66]) or
  at 1.38 times on MuSiQue (+1.53 [+0.79, +2.23]) is reading the ALL-only column; the decision is the user's.
- **What it leaves unread.** (a) Every number is on the current links and is provisional (section 10); the link set is not final, and PL8, the one pool re-run on it, is declared only after it is. (b) The e3v lists are not read here (this page declares none). (c) MetaQA's ladder stops at 3,500 (the steps of L below PL6's dev nodes), so its knee between 3,500 and PL6's 4,973 nodes
  is not read. (d) The cut ranks by cosine only; a parent-conditioned rank, a hop-weighted schedule and label-ordered caps are named in PL5 section 12 and are not in this round.

### 14.8 Surprises and contradictions with filed numbers

- **WebQSP's PL6-comparison status at 2,500 reads ABOVE although ALL is -6.85 points** [-8.25, -5.72]: neither pick reaches 0.95 on eval, and the nodes ratio 0.053 has its upper end below 1 (clause ABOVE 2). It is a price statement; the ALL-only status is BELOW. The same comparison reads BELOW on MuSiQue and MetaQA, where PL6's pick reaches 0.95 and the 2,500 pick does not (clause BELOW 1).
- **Eval ALL is not monotone in the ceiling**: MuSiQue's pick at 5,000 reads 0.8990 at 3,481 nodes against 0.8999 at 3,449 nodes for the 3,500 ceiling (dev 0.9081 against 0.9074): the dev frontier of all cells has no step between 3,518 and 5,783 nodes (section 1 fact 3, with the PL7 cells in), so the step buys nothing, and the dev gain of +0.07 points reads -0.08 on eval (point estimates; no interval is filed for this pair).
  WebQSP's 10,000 and 20,000 picks are the cells of its 7,000 and 14,000 for the same reason.
- **A ceiling on the mean is not a ceiling on a question**: MetaQA's 3,500 pick has a mean of 3,475 nodes and a maximum of 4,044; only the hard-cut cells have a pool at most K by construction, and no hard-cut cell of PL7 is a pick at any ceiling (14.5).
- **The declared host-cost estimate was too high for WebQSP, outside the stated factor of 1.5**: 4,566 s per part measured against about 9,000 declared (a factor of 2.0); MuSiQue 7,967 s per part against about 9,500; MetaQA 408 s per part against under an hour. The estimate came from PL6's seconds (section 11); it changes no number.
- **The declaration's pointer to section 13 for the results was wrong** (see the introduction of this section); nothing else in the declaration is corrected here.

### 14.9 The next changes (names only; none is adopted)

Declared separately, before any number, when the user says the link set is final: **PL8**, one pool re-run on it, in two stages (a dev-question subset to diagnose which golds the capped pools still miss, then the full dev and the one confirmation). From PL5 section 12, unchanged by these results: a parent-conditioned rank (a node's cosine plus its best parent's), a hop-weighted schedule,
label-ordered caps once a UL2 layout exists on every dataset. The link-side stage (section 10) comes first.

### 14.10 Disclosed and undone

- **`zpl5r.py` gained the `--pl7` flag of `miss`** after the call was read and before D1 and D2 were run (it reads `outputs/pl7/<d>/select.json` at `picks["2500"]["R95_all"]` and writes `outputs/pl7/<d>/miss.json`); the PL5 and PL6 paths of the script are unchanged. D1 and D2 ran at the three SHORT datasets' 2,500 picks, as section 7 declared, and equal PL5's records (14.6).
- **`select` ran on all seven datasets (26 Nov 03:04 to 03:05 IST) before the confirmation rows were issued**; `confirm` needed four cells that PL5's and PL6's confirmation grids do not hold (MetaQA 1, MuSiQue 2, WebQSP 1), the other eval columns were read from them. `call` ran once, after the nine confirmation parts arrived.
- **Every ceiling above 2,500 is a reading on the way down**, filed with its interval and never carried (section 6); the floor flag `at_floor` is my definition and flagged no pick on any dataset, so it changed no reading.
- **507 (MuSiQue), 819 (WebQSP) and 117 (MetaQA) new cells are added to the 6,400 or so per dataset in the selection** (1,443 in all; section 13 of the declaration said about 1,750, an estimate made before the lists were counted): a winner's dev ALL is optimistic, so every call is a paired interval on eval; the dev numbers of 14.3 and 14.5 are the selection's, not a result.
- **The pools are on the current links and provisional** (section 10); the four datasets inside 2,500 have no PL7 cells and no host row (section 5), their eval record is PL5's confirmation of the same cell.
- The grid arrays (`*.npz`) are not committed (as PL2's to PL6's); the heads, selects, diagnostics, the call and the builder are. The laptop select chain is `outputs/pl7/run_select.sh`.
