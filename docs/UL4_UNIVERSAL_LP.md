# UL4: one universal rule as a linear program (bounds and costs of 95 percent ALL-gold near 2,500 nodes)

Declared 26 November 2026, 03:39 IST (22:09 UTC the day before), before any UL4 number exists. **Nothing here trains, no model reads anything, no LLM or spaCy is used, no filed graph, pool, fit or record changes, and no link is removed from any store.** It reads records that are already filed (UL3's cell columns, PL2 to PL7's dev grids) and solves small linear programs on them on the laptop. It carries nothing downstream; it says what a universal rule can cost, and what no rule on these cells can do.

## 0. The user's words, and the rulings that bind

- 26 Nov, 03:3x: "No universal one-hop rule exists. we need to make it universal, tell the bounds or costs for that to happen i told you my limits, 95+ all coverage, as close as possible to 2.5k nodes / treat this as a linear programming problem to find the best values for the variables we have to optimize this constraint".
- Earlier and still binding: "we need to have universal rules" (one rule on all seven, carried everywhere or nowhere); S is the structural links, R the rule links, K the kNN, never conflated, "we will keep all weights"; the hub line is CLOSED (no degree, IDF or hub-key attribute appears anywhere below); no LLM, no spaCy, no learning; declare before any number; selection on dev, grading on eval; subset first.
- 25 Nov (coverage target): "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts"; 2,500 nodes is the bound, liftable only under one rule on all seven (PL6 lifted it as analysis; PL7 restricts back).

## 1. What I had read before this page

UL3's section 14 (the nine-cell by three-variant table of reach and price at RHO 0.95, aggregated on s1eval), PL6's section 13 (every dataset's U95 pick and its eval ALL and nodes) and PL7's declaration. **Of the UL4 quantities I have seen none, except one cell's dev and eval columns (MuSiQue, variant C, class R, bin 0: dev 76,365 candidates and 43 covered slots, eval 71,142 and 50) printed while checking the record's shape.** The nine PL7 confirmation grids were fetched but not read; the dev grids are read only by the code below.

## 2. Why a linear program, and what is honestly linear

A pool rule has two kinds of variables, and each gives one program.

- **L (links).** Which cells of the link set to keep. A cell is a class of pairs; keeping a fraction x of a cell keeps x of its candidates and x of its covered slots in expectation, so reach and price are linear in x. x in [0, 1] is the relaxation (a randomised keep); the rule that can be carried has x in {0, 1} (an integer program over the same constraints, solved too). The relaxation is a **lower bound** on the price of any rule on these cells.
- **Q (nodes per question).** How deep the pool goes for each question. Choosing a level k for each question q gives ALL (the share of questions whose pool holds every gold) and the mean nodes, both linear in the assignment x_qk. The 2,500 bound is a **mean** over questions in PL2 to PL7's own measures (`candidates_mean`), so a rule that spends 400 nodes on an easy question and 9,000 on a hard one is a rule on the same footing.

What is not linear, said first: the coverage of a pool is not additive across families or hops, so there is no LP on "quotas per family"; and links and pools interact (a tightened link set changes every pool). **L and Q are solved apart here; their joint effect is what PL8 measures after the link set is final.**

## 3. Layer L, the shared link keep-rule

**Data.** UL3's `outputs/ul3/<d>/cells.json`, the nine judged cells j (MultiHop-RAG RK GP; MuSiQue RK GP and CH; MetaQA SRK GP; WebQSP SRK GP; HotpotQA RK GP and SRK GP; 2Wiki RK GP and SRK GP). Per cell the dev ("sel") and eval columns: candidates and covered slots. **Shared keys (the same key in every dataset, defined by a rule, no learning):** K1 = the S > R > K group times the dot quintile (UL3's C variant, 15 keys); K2 = the label profile over the five common labels (S, R.mention, R.name_comention, R.title_comention, K.knn) times the dot quintile inside the exact profile (UL3's PC variant, at most 155 keys; the profile's tokens are already the same names in all nine cells). A key absent from a judged cell has zero columns there.

**Variables** x_c in [0, 1] (LP) or {0, 1} (ILP), one per shared key, the same in every judged cell.

**Constraints and objective, on the dev columns only.** reach_j(x) = sum_c x_c cov_jc / sum_c cov_jc, price_j(x) = sum_c x_c cand_jc / sum_c cand_jc.
- **LP-U(rho):** minimise tau subject to reach_j >= rho and price_j <= tau for every judged cell j. rho in {0.80, 0.90, 0.95, 0.99}; the call is rho = 0.95 (the PL target). **A.1 (S never removed) is the call**: x_c = 1 for every key whose class holds S; the same program with S free is solved beside.
- **LP-D_j(rho):** the same for one judged cell alone (private x): each cell's own optimum, which is the greedy cell order of UL3 made fractional. **Cost of universality** = tau*(LP-U) minus the largest tau*(LP-D_j).
- **Frontier (the cost in reach):** maximise r subject to reach_j >= r and price_j <= tau, tau in {0.30, 0.40, 0.50, 0.60, 0.80}.
- **Duals.** Which reach constraints bind at the call, and tau*(rho) at rho of 0.94, 0.95 and 0.96 (the price of a point of reach).

**The call (L).** The ILP-U x (binary, S protected, rho = 0.95, solved on dev) is graded once on eval's columns (point values, as UL3: no per-question columns are stored, so no interval): **UNIVERSAL_EXISTS** iff every judged cell has reach retained >= 0.92 and price retained <= 0.50 (UL3's thresholds, fixed before); **NOT_EXISTS** otherwise, with tau* (the bound no rule on these keys can beat on dev), the binding cells and the cost of A.1. K1 and K2 are two spaces; each is read, and the call is made for each (K2 contains K1's rules).

## 4. Layer Q, the depth allocation (dev; an oracle bound, never a method)

**Data.** The dev grids of PL2 to PL7 on the laptop (`outputs/pl<r>/<d>/grid_dev_frozen*.npz`: per question and cell the golds present and the pool size; `gold_counts`): all seven datasets. **The family of a dataset** is the cells named P or P+top{M}, where P is the base+walk of PL6's U95 pick (its filed name, not chosen here): MuSiQue `rrf1000_b400+FULL:h3_bf_v20000~cs-fu`, MetaQA `rrf1000_b25+STRUCT:h3_c100_v5000`, WebQSP `rrf1000_b400+FULL:h3_bf_v40000~cs-vr`, 2Wiki `rrf1000_b400+FULL:h1_c100~s20`, HotpotQA `rrf1000_b25+STRUCT:h1_c100~cs`, MultiHop-RAG `rrf1000_b25+FULL:h1_c100`, SQuAD `rrf1000_b15`. The cuts +top{M} are prefixes of one ranked list, so the family is a nested ladder of depths. a_qk = 1 iff present = gold_counts (> 0), n_qk = the pool size.

**Variables** x_qk >= 0, sum_k x_qk = 1 for each question.
- **Q-min:** minimise the mean of n subject to ALL = mean of a >= rho (rho = 0.95; 0.98 beside).
- **Q-cap(N):** maximise ALL subject to mean nodes <= N, N in {2,500; 3,500; 5,000; 7,000; 10,000}.
- **Beside:** the uniform ladder (one level for every question: the family's own table), its convex hull (a randomised uniform rule), and the oracle's gap to both; the share of questions whose cheapest covering level is at most 2,500 nodes; the share no level of the family covers (the family's ceiling on ALL); the shadow price (mean nodes per point of ALL at 0.95: the slope of the frontier between rho = 0.94 and 0.96).
- **Optimistic bound beside, labelled so:** the same programs over every cell of the dataset's table (about 6,000 cells), where a question may pick a lucky small cell. It is an upper limit on what any menu-based router could do, not a number to weigh.

**What it says and does not.** The oracle reads the labels, so Q-min is the **least** mean nodes any rule that chooses a depth per question from this family could use at 0.95 on dev; a real rule needs a signal that predicts the depth (not tested here). A dataset is **FEASIBLE_AT_2500** iff Q-min <= 2,500 on dev; else **INFEASIBLE_AT_2500** with Q-min and Q-cap(2,500). **No eval number is read in layer Q.**

## 5. Expectations, stated before any number (a failed one is reported, not explained away)

- **L1:** LP-U under A.1 at rho = 0.95 has tau* >= 0.80 on K1 and on K2 (MuSiQue's cells are nearly flat, and MetaQA's S cells alone hold 72 percent of its candidates). With S free tau* is lower, between 0.55 and 0.80. UNIVERSAL_EXISTS on neither space.
- **L2:** the cost of universality is under 0.10 on K2 and larger on K1 (K1 has 15 keys to serve nine cells).
- **Q1:** MetaQA is FEASIBLE_AT_2500 (its 1-hop questions need few nodes). MuSiQue's Q-min is above 2,500 but well under the uniform pick's 15,400. WebQSP's Q-min is above 2,500 by a large factor (its family reaches 0.95 on dev only at V = 40,000, and the oracle may save a half or less).
- **Q2:** the four datasets that already sit inside the bound (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki) are FEASIBLE_AT_2500, with Q-min below their uniform picks (50, 15, 78, 917 nodes).
- **What would surprise me:** a Q-min within 2,500 on MuSiQue or WebQSP. It would say the 2,500 bound is a depth-allocation problem and not a link or walk problem.

## 6. Gates (a failure stops that dataset's reading; none of its numbers is read)

- **G4-1 (L):** per judged cell and variant, the key columns sum to UL3's any-link totals on dev and eval (UL3's G3-1, re-read from the record).
- **G4-2 (L):** LP-D_j with S free at rho = 0.95 on K1 has a dev price at most UL3's filed integer keep-set price on dev for C at RHO 0.95 (the fractional optimum cannot be above the greedy integer one); the two are printed.
- **G4-3 (Q):** a cell present in more than one grid file equals itself element for element (present and size, every dev question); the question order (`qids`) and `gold_counts` agree across parts; the question count equals the dev carve's filed count.
- **G4-4 (Q):** the family's cells reproduce the filed dev ALL and mean nodes of PL6's select (`outputs/pl6/<d>/select.json`; PL5's for the four small datasets) to 1e-9.
- **G4-5:** the self-test: the LP solver equals brute force on toy instances (L: all binary x on 8 keys and 3 cells; Q: all level assignments on 6 questions and 3 levels); the frontier equals the Lagrangian sweep; a deliberately wrong objective is caught.

## 7. Code, outputs

`outputs/mp_unified/zul4lp.py` (new, mine; numpy and scipy only; reads UL3's `cells.json` and the PL dev grids and edits none of them). `selftest`, `lp-links`, `lp-depth`, `report`. Outputs only under `outputs/ul4/` (json and md; no arrays). Laptop, one process, no host row.

## 8. What follows

The numbers go into a results section of this page, then to the user with the bounds and the costs. Nothing is carried. If L says NOT_EXISTS, the link set stays the base layout (UL2 a70ec8cc) and the bound says what a universal rule on these keys costs; any new link rule is a new page. If Q says INFEASIBLE_AT_2500 on a dataset, the gap to the uniform pick says what a depth signal could buy and the Q-min says how far the 2,500 bound is from reach under these cells. The PL agent is told only what the user decides about the link set.

## 9. Disclosed

- The key spaces K1 and K2, rho, tau, N, the families (PL6's picks), the thresholds (0.92, 0.50, from UL3) and the calls were fixed before any UL4 number; I had read UL3's and PL6's tables, which are what suggested a link filter with a floor and a pool bound that three datasets miss.
- The program L reads the same cell columns UL3 read and re-optimises over them: it cannot find a rule that UL3's cells cannot express (a path-level rule, a new attribute); it prices the ones they can.
- Layer Q uses the dev labels to choose a depth per question: an oracle, labelled so wherever it is printed.

## 10. Results (26 November 2026, 03:51 IST; 22:21 UTC)

Everything below is read from `outputs/ul4/links.json`, `depth.json` and `report.md` (script sha256 `6fb3258dcb9953a1`, `zul4lp.py`, committed with this section). Nothing was read from eval in layer Q; layer L's integer rule was solved on dev and graded once on eval (point values). Nothing is carried and no link, pool or fit changes.

### 10.1 Gates, and two disclosures

- **G4-1 PASS** (the code stops otherwise): for all nine judged cells, both variants, both carves, the key columns sum to the any-link totals.
- **G4-2 PASS**: every judged cell's fractional own optimum (S free, rho 0.95) is at most UL3's filed integer keep-set price on dev, in K1 and in K2 (the table of both is in `links.json`, `g4_2`).
- **G4-3 PASS**: the parts of every grid tile the dev questions, the question order and gold counts agree across grids, every cell held by two grids is equal element for element, and the dev question count equals PL6's.
- **G4-4 PASS**: for every family cell that PL6's select table lists, the dev ALL and mean nodes agree to 1e-9 (cells checked: multihoprag 8, squad 1, hotpotqa 7, 2wiki 7, musique 4, metaqa 8, webqsp 4). The ladder's nesting held (a larger cut never held fewer golds or fewer nodes for any question; violations: 0).
- **G4-5 PASS**: the solver equals brute force on toy instances (links: all binary x on 8 keys and 3 cells, S free and S protected; depth: all level assignments on 6 questions and 3 levels), the depth knapsack equals linprog on the assignment variables, and the hull is right.
- **Disclosure 1 (the frontier check and a weak wrong-objective check).** The declaration lists a check that the frontier equals the Lagrangian sweep and that a deliberately wrong objective is caught. The self-test first passed without the frontier check; I added it (the frontier's reach, solved the other way, returns its price cap) and it passes. The wrong-objective check is weak (it asserts only that maximising the price is not the minimum).
- **Disclosure 2 (the first depth run used the wrong family).** My first depth run took the family from PL6's select table, which lists only some of the cuts; the declaration (4) says every grid cell named P or P+top{M}, and the grids hold a denser ladder. I had printed the first run before I saw the difference: MuSiQue 6,031, MetaQA 3,394, WebQSP 15,899 mean nodes (the table's smallest cut was 5,000 and 10,000 on MuSiQue and WebQSP, so nothing under 5,400 and 10,399 nodes could appear). I corrected the code to the declared definition and re-ran everything; the first run is kept as `outputs/ul4/depth_first_run_table_cells.json`. The numbers below are the declared family's.

### 10.2 Layer L: the shared link keep-rule

#### Key space K1 (15 keys)

| program | rho | worst price tau* | largest own optimum | cost of universality | binding reach cells |
|---|---|---|---|---|---|
| S protected (A.1, the call) | 0.80 | 90.6 | 80.6 | 10.0 | musique/RK/CH, hotpotqa/RK/GP |
| S protected (A.1, the call) | 0.90 | 94.1 | 81.9 | 12.2 | musique/RK/CH |
| S protected (A.1, the call) | 0.95 | 97.0 | 82.6 | 14.4 | musique/RK/CH |
| S protected (A.1, the call) | 0.99 | 99.3 | 95.4 | 4.0 | musique/RK/CH, metaqa/SRK/GP, hotpotqa/RK/GP |
| S free | 0.80 | 74.0 | 49.6 | 24.4 | musique/RK/CH, 2wiki/SRK/GP |
| S free | 0.90 | 85.6 | 63.3 | 22.2 | musique/RK/CH, hotpotqa/SRK/GP, 2wiki/SRK/GP |
| S free | 0.95 | 92.7 | 78.7 | 14.0 | musique/RK/CH, hotpotqa/SRK/GP, 2wiki/SRK/GP |
| S free | 0.99 | 98.5 | 95.4 | 3.1 | musique/RK/CH, metaqa/SRK/GP, hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/SRK/GP |

The price of one point of reach near the call (tau*(0.96) - tau*(0.94)) / 2, in points of price: S protected 0.6; S free 1.4.

**Each judged cell's own optimum at rho 0.95 (dev price; the least a rule made for that cell alone could pay), and the S floor (the price of S alone):**

| judged cell | own optimum, S free | own optimum, S protected | S floor (dev) | S floor (eval) | S-alone reach (dev) |
|---|---|---|---|---|---|
| multihoprag/RK/GP | 41.0 | 41.0 | 0.0 | 0.0 | 0.0 |
| musique/RK/GP | 78.5 | 78.5 | 0.0 | 0.0 | 0.0 |
| musique/RK/CH | 78.7 | 78.7 | 0.0 | 0.0 | 0.0 |
| metaqa/SRK/GP | 17.1 | 82.6 | 71.6 | 72.3 | 4.1 |
| webqsp/SRK/GP | 25.1 | 52.3 | 36.9 | 38.5 | 36.2 |
| hotpotqa/RK/GP | 74.8 | 74.8 | 0.0 | 0.0 | 0.0 |
| hotpotqa/SRK/GP | 46.7 | 53.8 | 53.8 | 52.1 | 99.7 |
| 2wiki/RK/GP | 58.5 | 58.5 | 0.0 | 0.0 | 0.0 |
| 2wiki/SRK/GP | 43.0 | 43.0 | 40.0 | 35.3 | 93.8 |

**The cost in reach of a price (max over x of the worst judged cell's dev reach, at price <= tau for every cell):**

| price cap tau | S protected | S free |
|---|---|---|
| 0.30 | infeasible | 35.1 |
| 0.40 | infeasible | 46.7 |
| 0.50 | infeasible | 57.7 |
| 0.60 | infeasible | 67.2 |
| 0.80 | 42.9 | 85.3 |

**The integer rule, S protected (the call), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 97.5; kept 12 of 15 keys; dropped: K|0, K|2, R|0).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 82.9 | 99.9 / 83.8 | OUT |
| musique/RK/GP | 95.4 / 81.6 | 97.2 / 86.4 | OUT |
| musique/RK/CH | 95.3 / 81.6 | 97.2 / 86.4 | OUT |
| metaqa/SRK/GP | 95.3 / 97.5 | 95.6 / 97.5 | OUT |
| webqsp/SRK/GP | 99.3 / 89.1 | 99.8 / 90.6 | OUT |
| hotpotqa/RK/GP | 95.8 / 79.4 | 96.2 / 62.6 | OUT |
| hotpotqa/SRK/GP | 99.9 / 90.2 | 100.0 / 80.9 | OUT |
| 2wiki/RK/GP | 99.8 / 79.9 | 99.8 / 78.8 | OUT |
| 2wiki/SRK/GP | 99.9 / 87.1 | 99.9 / 86.6 | OUT |

**The integer rule, S free (beside), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 97.5; kept 12 of 15 keys; dropped: K|0, K|2, R|0).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 82.9 | 99.9 / 83.8 | OUT |
| musique/RK/GP | 95.4 / 81.6 | 97.2 / 86.4 | OUT |
| musique/RK/CH | 95.3 / 81.6 | 97.2 / 86.4 | OUT |
| metaqa/SRK/GP | 95.3 / 97.5 | 95.6 / 97.5 | OUT |
| webqsp/SRK/GP | 99.3 / 89.1 | 99.8 / 90.6 | OUT |
| hotpotqa/RK/GP | 95.8 / 79.4 | 96.2 / 62.6 | OUT |
| hotpotqa/SRK/GP | 99.9 / 90.2 | 100.0 / 80.9 | OUT |
| 2wiki/RK/GP | 99.8 / 79.9 | 99.8 / 78.8 | OUT |
| 2wiki/SRK/GP | 99.9 / 87.1 | 99.9 / 86.6 | OUT |

#### Key space K2 (115 keys)

| program | rho | worst price tau* | largest own optimum | cost of universality | binding reach cells |
|---|---|---|---|---|---|
| S protected (A.1, the call) | 0.80 | 84.1 | 80.9 | 3.1 | hotpotqa/RK/GP, 2wiki/RK/GP |
| S protected (A.1, the call) | 0.90 | 91.1 | 82.4 | 8.7 | musique/RK/GP, hotpotqa/RK/GP |
| S protected (A.1, the call) | 0.95 | 95.0 | 83.1 | 12.0 | musique/RK/GP, hotpotqa/RK/GP |
| S protected (A.1, the call) | 0.99 | 98.8 | 88.8 | 10.0 | musique/RK/GP, hotpotqa/RK/GP |
| S free | 0.80 | 55.1 | 34.4 | 20.7 | musique/RK/CH |
| S free | 0.90 | 73.6 | 55.4 | 18.2 | musique/RK/CH, webqsp/SRK/GP, hotpotqa/SRK/GP, 2wiki/SRK/GP |
| S free | 0.95 | 85.8 | 71.5 | 14.3 | musique/RK/CH, webqsp/SRK/GP, 2wiki/SRK/GP |
| S free | 0.99 | 96.9 | 88.6 | 8.3 | musique/RK/GP, webqsp/SRK/GP, hotpotqa/RK/GP, 2wiki/SRK/GP |

The price of one point of reach near the call (tau*(0.96) - tau*(0.94)) / 2, in points of price: S protected 0.8; S free 2.6.

**Each judged cell's own optimum at rho 0.95 (dev price; the least a rule made for that cell alone could pay), and the S floor (the price of S alone):**

| judged cell | own optimum, S free | own optimum, S protected | S floor (dev) | S floor (eval) | S-alone reach (dev) |
|---|---|---|---|---|---|
| multihoprag/RK/GP | 34.6 | 34.6 | 0.0 | 0.0 | 0.0 |
| musique/RK/GP | 71.5 | 71.5 | 0.0 | 0.0 | 0.0 |
| musique/RK/CH | 71.1 | 71.1 | 0.0 | 0.0 | 0.0 |
| metaqa/SRK/GP | 17.4 | 83.1 | 71.6 | 72.3 | 4.1 |
| webqsp/SRK/GP | 25.2 | 52.2 | 36.9 | 38.5 | 36.2 |
| hotpotqa/RK/GP | 64.7 | 64.7 | 0.0 | 0.0 | 0.0 |
| hotpotqa/SRK/GP | 43.9 | 53.8 | 53.8 | 52.1 | 99.7 |
| 2wiki/RK/GP | 42.6 | 42.6 | 0.0 | 0.0 | 0.0 |
| 2wiki/SRK/GP | 37.8 | 40.3 | 40.0 | 35.3 | 93.8 |

**The cost in reach of a price (max over x of the worst judged cell's dev reach, at price <= tau for every cell):**

| price cap tau | S protected | S free |
|---|---|---|
| 0.30 | infeasible | 58.8 |
| 0.40 | infeasible | 67.8 |
| 0.50 | infeasible | 76.1 |
| 0.60 | infeasible | 83.6 |
| 0.80 | 72.3 | 92.7 |

**The integer rule, S protected (the call), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 97.5; kept 102 of 115 keys; dropped: K.knn|0, K.knn|2, R.mention+K.knn|2, R.mention+K.knn|3, R.mention|0, R.name_comention|0, R.title_comention+K.knn|0, R.title_comention+K.knn|1, R.title_comention+K.knn|2, R.title_comention+K.knn|3, R.title_comention+K.knn|4, R.title_comention|0, R.title_comention|1).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 83.6 | 99.9 / 84.4 | OUT |
| musique/RK/GP | 95.5 / 82.1 | 96.5 / 86.0 | OUT |
| musique/RK/CH | 95.4 / 82.1 | 96.5 / 86.0 | OUT |
| metaqa/SRK/GP | 95.2 / 97.5 | 95.5 / 97.6 | OUT |
| webqsp/SRK/GP | 98.8 / 89.1 | 98.7 / 90.5 | OUT |
| hotpotqa/RK/GP | 96.1 / 80.0 | 95.8 / 63.2 | OUT |
| hotpotqa/SRK/GP | 99.9 / 90.2 | 100.0 / 81.2 | OUT |
| 2wiki/RK/GP | 98.5 / 81.3 | 98.7 / 80.6 | OUT |
| 2wiki/SRK/GP | 99.9 / 87.4 | 99.8 / 86.9 | OUT |

**The integer rule, S free (beside), solved on dev at rho 0.95: NOT_EXISTS** (dev worst price 92.4; kept 98 of 115 keys; dropped: K.knn|0, K.knn|2, R.mention+K.knn|2, R.mention+K.knn|3, R.mention|0, R.name_comention|0, R.title_comention+K.knn|2, R.title_comention+K.knn|3, R.title_comention+K.knn|4, R.title_comention|0, S+R.mention+K.knn|2, S+R.mention+K.knn|3, S+R.mention+K.knn|4, S+R.mention|0, S+R.title_comention+K.knn|2, S+R.title_comention+K.knn|3, S+R.title_comention+K.knn|4).

| judged cell | dev reach / price | eval reach / price | SCREEN_IN |
|---|---|---|---|
| multihoprag/RK/GP | 99.8 / 83.6 | 99.9 / 84.5 | OUT |
| musique/RK/GP | 95.5 / 82.2 | 96.8 / 86.1 | OUT |
| musique/RK/CH | 95.4 / 82.2 | 96.8 / 86.1 | OUT |
| metaqa/SRK/GP | 95.2 / 92.4 | 95.5 / 92.6 | OUT |
| webqsp/SRK/GP | 97.4 / 78.2 | 96.3 / 79.0 | OUT |
| hotpotqa/RK/GP | 96.1 / 80.1 | 95.9 / 63.2 | OUT |
| hotpotqa/SRK/GP | 95.5 / 82.2 | 95.1 / 69.5 | OUT |
| 2wiki/RK/GP | 98.7 / 81.4 | 98.7 / 80.7 | OUT |
| 2wiki/SRK/GP | 95.3 / 81.7 | 95.8 / 82.2 | OUT |

### 10.3 Layer Q: the depth allocation

Mean nodes per question at ALL >= 0.95 on the dev carve, by rule. **Uniform** = one level for every question (the family's own table); **hull** = a randomised uniform rule; **oracle depth** = each question gets the cheapest level that holds all its golds (the LP optimum; it reads the golds).

| dataset | family cells | uniform (least nodes at 0.95) | hull | oracle depth (family) | oracle ALL at 2,500 | uniform ALL at 2,500 | family ceiling on ALL | verdict at 2,500 |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 8 | 50 | 50 | 50 | 99.6 | 99.6 | 99.6 | FEASIBLE_AT_2500 |
| squad | 1 | 15 | 15 | 15 | 95.2 | 95.2 | 95.2 | FEASIBLE_AT_2500 |
| hotpotqa | 7 | 77 | 65 | 64 | 96.8 | 96.8 | 96.8 | FEASIBLE_AT_2500 |
| 2wiki | 7 | 908 | 899 | 561 | 95.5 | 95.5 | 95.5 | FEASIBLE_AT_2500 |
| musique | 10 | 15,400 | 14,660 | 3,382 | 78.4 | 78.4 | 96.3 | INFEASIBLE_AT_2500 |
| metaqa | 10 | 4,984 | 4,921 | 2,804 | 88.9 | 46.4 | 96.1 | INFEASIBLE_AT_2500 |
| webqsp | 12 | 39,572 | 38,940 | 10,250 | 55.5 | 55.5 | 95.6 | INFEASIBLE_AT_2500 |

**The menu each program had (the family's levels, its smallest level's mean nodes, and whether the nesting held):**

| dataset | levels | smallest level (mean nodes) | questions where a larger cut held less | G4-4 cells checked |
|---|---|---|---|---|
| multihoprag | 8 | 50 | 0 | 8 |
| squad | 1 | 15 | 0 | 1 |
| hotpotqa | 7 | 64 | 0 | 7 |
| 2wiki | 7 | 450 | 0 | 7 |
| musique | 10 | 2,500 | 0 | 4 |
| metaqa | 10 | 50 | 0 | 8 |
| webqsp | 12 | 2,500 | 0 | 4 |

**ALL (percent) at a mean-node cap, oracle depth in the family (uniform in brackets):**

| dataset | 2,500 | 3,500 | 5,000 | 7,000 | 10,000 |
|---|---|---|---|---|---|
| multihoprag | 99.6 (99.6) | 99.6 (99.6) | 99.6 (99.6) | 99.6 (99.6) | 99.6 (99.6) |
| squad | 95.2 (95.2) | 95.2 (95.2) | 95.2 (95.2) | 95.2 (95.2) | 95.2 (95.2) |
| hotpotqa | 96.8 (96.8) | 96.8 (96.8) | 96.8 (96.8) | 96.8 (96.8) | 96.8 (96.8) |
| 2wiki | 95.5 (95.5) | 95.5 (95.5) | 95.5 (95.5) | 95.5 (95.5) | 95.5 (95.5) |
| musique | 78.4 (78.4) | 95.8 (81.9) | 96.3 (84.5) | 96.3 (88.4) | 96.3 (91.7) |
| metaqa | 88.9 (46.4) | 96.1 (63.1) | 96.1 (96.1) | 96.1 (96.1) | 96.1 (96.1) |
| webqsp | 55.5 (55.5) | 73.3 (58.7) | 80.5 (62.2) | 86.4 (65.9) | 94.3 (69.5) |

**Share of questions whose cheapest covering pool in the family is at most N nodes (percent):**

| dataset | 2,500 | 3,500 | 5,000 | 7,000 | 10,000 | covered by no level |
|---|---|---|---|---|---|---|
| multihoprag | 99.6 | 99.6 | 99.6 | 99.6 | 99.6 | 0.4 |
| squad | 95.2 | 95.2 | 95.2 | 95.2 | 95.2 | 4.8 |
| hotpotqa | 96.8 | 96.8 | 96.8 | 96.8 | 96.8 | 3.2 |
| 2wiki | 95.5 | 95.5 | 95.5 | 95.5 | 95.5 | 4.5 |
| musique | 78.4 | 81.9 | 84.5 | 88.4 | 91.7 | 3.7 |
| metaqa | 46.4 | 63.1 | 63.6 | 96.1 | 96.1 | 3.9 |
| webqsp | 55.5 | 58.7 | 62.2 | 65.9 | 69.5 | 4.4 |

**The price of ALL, and the optimistic bound beside.** Nodes per point of ALL at 0.95 (oracle depth, family) and the optimistic all-cell bound (a question may pick a lucky small cell of ~6,000):

| dataset | nodes per point of ALL | oracle depth at 0.98 (family) | all-cell oracle at 0.95 (optimistic) | all-cell ALL at 2,500 (optimistic) |
|---|---|---|---|---|
| multihoprag | 0 | 50 | 11 | 100.0 |
| squad | - | - | 5 | 100.0 |
| hotpotqa | 0 | - | 20 | 99.9 |
| 2wiki | - | - | 42 | 98.6 |
| musique | 137 | - | 133 | 98.7 |
| metaqa | 50 | - | 485 | 98.7 |
| webqsp | - | - | 185 | 97.0 |

### 10.4 Expectations against the outcome (5; a failed one is reported, not explained away)

- **L1** (A.1: tau* >= 0.80 on K1 and K2): **held** (K1 97.0, K2 95.0). (S free between 0.55 and 0.80): **FAILED** (K1 92.7, K2 85.8: higher than I wrote). (UNIVERSAL_EXISTS on neither space): **held** (NOT_EXISTS on K1 and K2, S protected and S free).
- **L2** (cost of universality under 0.10 on K2): **FAILED** (K2 12.0 points; K1 14.4). (larger on K1 than K2): **held**, by 2.4 points.
- **Q1** (MetaQA FEASIBLE_AT_2500): **FAILED** (Q-min 2,804 nodes, 1.12 times 2,500; ALL at 2,500 88.9). (MuSiQue above 2,500 and well under 15,400): **held** (3,382). (WebQSP far above 2,500, the oracle saving a half or less of the uniform pick): **FAILED** (10,250 nodes, 4.1 times 2,500; it saves 74.1 percent of the uniform pick, more than half).
- **Q2** (the four small datasets FEASIBLE_AT_2500): **held**. (Q-min below their uniform picks): held on hotpotqa, 2wiki; equal on the others (one level, or the cheapest level already holds the ceiling).
- **The surprise** (Q-min within 2,500 on MuSiQue or WebQSP): **did not happen.**

### 10.5 What this says, and what it does not

**Links (layer L).** On the keys UL3's cells can express (the S > R > K group, the label profile, the dot quintile), no shared keep-rule exists, even as a fractional relaxation, even with S free. To keep 95 percent of the covered gold pairs in the worst of the nine judged cells, a universal rule must retain **95.0 percent of the candidates with S never removed (K2; 97.0 on K1)** and 85.8 percent with S free (K2; 92.7 on K1); the filing's ceiling was 50. The other way round: at a price cap of 50 percent no S-protected rule exists at all (the S keys alone hold 72 percent of MetaQA's candidates), and with S free the worst cell keeps 76.1 percent of its reach (K2; 57.7 on K1). A point of reach costs 2.6 points of price with S free and 0.8 with S protected (K2). Two things are priced separately: **S never removed** costs MetaQA 65 points (its own optimum is 17 percent with S free and 83 percent with S protected; S holds 72 percent of its candidates and 4 percent of its covered slots) and WebQSP 27; **universality** costs 12 points (K2) on top of the worst single cell. MuSiQue (own optimum 71 to 79 percent) and HotpotQA without S (65 to 75) are the cells that bind: their gold pairs are hardly separated from the other candidates by label and dot, so any rule that serves them keeps nearly everything. A one-hop link filter on these keys therefore cannot be the lever for the node bound: even the S-free, fractional optimum removes at most about one candidate in seven (K2; one in fourteen on K1).

**Depth (layer Q).** With every question given the cheapest level of its dataset's own ladder that holds all its golds (an oracle: it reads the golds), the mean nodes at ALL 0.95 on dev are **MuSiQue 3,382** (uniform 15,400), **MetaQA 2,804** (uniform 4,984), **WebQSP 10,250** (uniform 39,572): 1.35, 1.12 and 4.1 times the 2,500 bound, and 4.6, 1.8 and 3.9 times under the uniform pick. The four small datasets already sit inside the bound. So the depth, and not the link filter, is where the factor lies: a depth signal as good as the oracle would put MetaQA and MuSiQue within about 1.1 to 1.4 times the bound and leave WebQSP at 4 times. The mean-over-questions form of the bound is what makes this possible: some questions need 50 nodes and others 10,000.

**What it does not say.** (1) The oracle reads the golds; a real rule needs a signal that predicts the depth from the question and the first stage, and none is tested here. (2) On MuSiQue and WebQSP the ladder's smallest level is 2,500 nodes (the walk was cut no lower in the grids), so Q-min is the least **on that ladder**; a ladder with smaller cuts can only lower it, so INFEASIBLE_AT_2500 on those two means 'not reached on the cuts that exist', not 'impossible'. The 'all cells' column is the opposite extreme (a best of about 6,000 cells per question; it selects on dev) and is not a bound anyone can use. (3) The two layers are solved apart: a tighter link set changes every pool, and the joint effect is PL8's to measure after the link set is final. (4) Layer L prices only what the cells can express; a rule on a new attribute, or on a path rather than a hop, is a new page.

### 10.6 What follows

- **Links.** The link set stays the base layout (UL2 a70ec8cc, UL3 a3a87915, now UL4). The bound says a label-and-dot keep-rule cannot get near 50 percent price at 95 percent reach; a different rule needs a different attribute or a path-level cell, which is the gold-pattern diagnosis (the user's 26 Nov message, to be done per dataset on the dev slice, subset first, declared before any number). PL8 is not declared: the user has not said the link set is final.
- **Depth.** The decision for the user is whether a question-side depth signal is worth a declared page: the oracle bound says the factor is there (1.8 to 4.5 times under the uniform pick on the three short datasets), and whether a statistic of the first-stage ranking or of the walk's own growth predicts the depth is not tested here (a threshold on one such statistic, chosen on dev by this same program, would be the form). Before that, the cheapest read is a dev-only ladder of smaller cuts for MuSiQue and WebQSP (prefixes of the same ranked list, no new walk), to see whether the 2,500 floor of their ladders hides a lower Q-min. Neither is declared here.
- **Nothing is carried.** No row, no host job, no model.

### 10.7 Disclosed

- The expectations L1, L2 (first half), Q1 (MetaQA, WebQSP's share) failed; they are reported as written above and not explained away.
- The key spaces, rho, tau, N, the family definition, the thresholds and the calls were fixed before any number; the first depth run's departure from the family definition is Disclosure 2.
- Layer L's relaxation is a lower bound on the price of any rule on these keys; the integer rule solved on dev (the call) has a dev price of 97.5 (K2) and 97.5 (K1), against the relaxation's 95.0 and 97.0.
- Eval values of the integer rule are point values (UL3's record has no per-question columns), so no interval is given; all nine judged cells fail the price test (eval price 62 to 98 percent against the 50 percent ceiling) by a margin far larger than any plausible interval.
