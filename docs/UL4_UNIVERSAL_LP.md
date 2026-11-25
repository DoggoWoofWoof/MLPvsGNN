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
