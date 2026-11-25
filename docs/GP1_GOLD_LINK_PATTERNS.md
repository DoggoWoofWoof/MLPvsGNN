# GP1: what separates the links that carry gold paths from the other links (a constrained program over shared attribute keys)

Declared 26 November 2026, 04:10 IST (22:40 UTC the day before), before any GP1 number exists. **Nothing here trains, no model reads anything, no LLM or spaCy is used, no filed graph, pool, fit or record changes, and no link is removed from any store.** GP1 reads the gold only to grade (it names which links lie on the shortest link paths from the first stage to the golds the first stage missed), measures what those links have that their neighbours do not, and solves one constrained program for one shared keep-rule. It carries nothing downstream. Eval is not read on this page.

## 0. The user's words, and the rulings that bind

- 26 Nov (a): "then we need to diagnose dataset by dataset and find a pattern from the gold links to make it more exact".
- 26 Nov (b): "No universal one-hop rule exists. we need to make it universal, tell the bounds or costs for that to happen i told you my limits, 95+ all coverage, as close as possible to 2.5k nodes".
- 26 Nov (c), the form: "instead of linear programming do it as a constrained programming problem". Section 6 states the variables, objective and constraints, names which solver is exact, and keeps the LP as the relaxation and the sensitivity.
- Earlier and still binding: "we need to have universal rules" (one rule on all seven, same parameters, carried everywhere or nowhere); S is the structural links, R the rule links, K the kNN, never conflated, **S never removed (A.1)**, "we will keep all weights"; **the hub line is CLOSED (UL2 8.15): no degree, IDF or hub-key attribute appears below**; no LLM, no spaCy, no learning; declare before any number; selection on dev, grading on eval; subset first, then full (the user's ~23:25 ruling of 25 Nov); "we will never write any notes".

## 1. What I had read before this page, and what I had not

Read in full: UL3 (declaration and section 14), UL4 (declaration and section 10, the tables of layers L and Q), U2A, U2J, U2K, UL2H's declaration and its report's MuSiQue and MetaQA tables, the engines `zul3a.py` (cells, keep rule), `zul2p.py` (anchors, B0 units) and `zul2h.py` (the walker). Read as headings and result tails only: U2L, U2I, U2G, UL2, UL2P, UL2S, UL1. **GP1 quantities I have seen: none.** What I have seen that bears on this page: UL2H's distance of the missed golds from the first stage's seeds in the any-link graph (MuSiQue s1eval, e3v seeds: 55.4 / 35.9 / 8.2 / 0.6 percent at 1 / 2 / 3 / more than 3 hops), which says the needed links sit at one to three hops; UL3's and UL4's finding that label and dot cannot separate gold pairs from the other pairs of an anchor (tau* 86 to 97 percent); U2J and U2K's mention-link features (lead, pos, count, caps, capmid, tokens, mutual, in- and out-degree), whose fitted weights against hyperlinks had tokens positive and in-degree negative. None of those numbers is a GP1 number, and in-degree is excluded here.

## 2. The question, and why path-level

UL3 asked, on one hop, whether a pair that is a gold pair differs by label and dot from the other pairs at the same anchor. It does not (price 36 to 100 percent at reach 95). Two things that screen could not see are: (1) a gold the first stage missed is reached through intermediate nodes, so the links that carry it are not gold pairs and one-hop precision never counts them; (2) the attributes were only label and dot. **GP1 asks: for the golds the first stage misses, take the links on the shortest link paths from the first-stage seeds to the gold (the needed links). Against the other links that leave the same nodes at the same step, which attributes of a link separate the two, per dataset, and does one shared rule on those attributes keep 95 percent of the needed links at a price that makes the 2,500-node bound reachable?** The per-dataset diagnosis (section 7) is the user's (a); the shared rule (section 6) is (b) and (c).

## 3. Population, seeds, graph, subset

- **Datasets and judged cells** (UL3's, plus SQuAD, which has one gold per question and so no gold pair but does have a path from the seeds to its gold): MultiHop-RAG RK; MuSiQue RK; SQuAD RK; MetaQA SRK; WebQSP SRK; HotpotQA RK and SRK; 2Wiki RK and SRK (ten judged cells, seven datasets). A dataset **passes** iff each of its cells does. If SQuAD has no dev carve in the package, it is NOT_READ and the universal read says six of seven, said first.
- **The graph** is UL2H's: the unit `any` = B0 = S (SRK) | R.mention | R.name_comention | R.title_comention | K.knn (the filed e3v layout, UL2 section 12 a70ec8cc), undirected, never capped. Graph and walker are UL2H's own (`zul2h.Comp`, `zul2h.Walker`, imported, not edited).
- **Seeds** = the dense top 5 U the SPLADE top 5 of the question (first occurrence kept, dense first; 5 to 10 distinct rows), as UL2H. **On the dev carve the lists are the package's** (`retrieval_cache/dense_top1000.npz` and `splade_top1000.npz` at the carve's rows; the lists PL's dev grids read). UL2H's e3v lists exist for s1eval only and the e3v query encoder is not run here (no GPU). MultiHop-RAG: PL1 S0a's lists on its dev split. This is disclosed: the seeds on dev are the frozen encoder's, the eval grade (if ever declared) would use e3v's.
- **Carves.** Selection data = the dev carve only: s1sel (WebQSP: PL's dev, the non-holdout half of train, `outputs/pl2/webqsp/pop_dev.npz`; MultiHop-RAG: its dev split). **Subset first (the user's ruling): the quarter Q = the dev questions whose index (position in the carve's id order after zero-gold questions are dropped) mod 4 = 0 selects; the rest R = index mod 4 != 0 grades.** Order of runs: the quarter is extracted and the program is solved on it; its record (the chosen attributes, directions, cut points and the solved keep set) is committed; **only then** R is extracted and the committed rule is graded on it. s1eval is not read on this page. A rule that passes on R would be graded once on eval on its own page.
- **Distance.** A gold in the seeds is found (distance 0). For a missed gold g, d(g) = its hop distance from the seeds in `any`, 1 to 3 (UL2H's walker: levels 1 and 2 outward, "within 3" from the gold's side); d = 4 or more is **unreached** and carries no needed link (its share is reported).

## 4. Needed links, controls, the unit of every count

- **Needed links N(g)** of a missed gold g at distance d: every link (a, c) of `any` that lies on some shortest path of length d from a seed to g (a at level l from the seeds, c at level l + 1, and c at distance d - l - 1 from g). A **needed unit** is (question, link) with the link's step l + 1 in {1, 2, 3}. The set is exact (the shortest-path DAG), except that a gold's units are capped at 5,000 (the first 5,000 in a fixed hash order of the link key; the number of capped golds is reported; if more than 5 percent of a dataset's reachable golds are capped, its read carries the flag CAPPED and its call is NOT_READ).
- **Weight.** Each gold has total weight 1, spread equally over its needed units: a unit's weight is the sum over the golds it serves of 1 / |N(g)|. So the weights sum to the number of reachable missed golds, and a hub gold with thousands of parallel paths does not outvote a gold with one path. The unweighted shares are printed beside.
- **Controls.** For every node a that is the source of a needed unit in question q at step l + 1: every frontier link (a, c) of `any` (c not at a level <= l from the seeds, i.e. a link a walk expanding from a would take) that is **not** a needed unit of q. A source's controls are sampled to at most 200 (fixed hash order of the link key) and each carries the weight w = (a's control count) / (the number sampled), so the weights sum to the exact count of frontier controls (gate G-G4); a question keeps at most 20,000 sampled controls (the levels share it in proportion). **The comparison is a link against the other links leaving the same source at the same step**, the user's and the lead's wording.
- **Price** of a rule is measured on this candidate set (needed distinct links, count 1 each, plus the weighted controls): the share kept. It is the price at the sources of needed links, not the pool size. The pool-size guard (the kept graph re-walked from the seeds over every question) is named in section 10 and is not run here.

## 5. The attributes (declared list; each extractable without an LLM or spaCy; none is a degree, IDF or hub key)

A link is traversed from a (source) to c (tail). "Text" attributes read the two nodes' text from the served package (`nodes.jsonl`); on a KB node the title is its text (the entity name), as `u1b.title_of`.

| id | attribute | bins in the program (at most 4, `na` = undefined for that link) |
|---|---|---|
| A1 | `grp` the S > R > K group of the link (the label that made it; the rule label, not the store) | S / R / K (always in the key) |
| A2 | `dot` the e3v dot of the two stored rows (UL3's) | three bins at shared cut points: the terciles of the pooled control links of the seven datasets' quarters, each dataset weighted equally; the cuts are written into the record before any solve |
| A3 | `rank` the rank of c among a's frontier links by dot (1 = highest) | 1 to 3 / 4 to 30 / above 30 |
| A4 | `votes` the number of distinct step-(l) nodes of the question's own ball that link to c (c's parents among the seeds' ball): a seed-set agreement statistic, question-conditioned and label-free | 1 / 2 / 3 or more. **Guard (fixed now):** its Spearman correlation with c's whole-graph neighbour count is printed per dataset; if it exceeds 0.70 on any dataset, A4 is a degree proxy and is dropped from the program (kept in the diagnosis) |
| A5 | `dir` who names whom: a's text names c's title (out), c's text names a's title (in), both (mutual), neither | mutual / one-way / none (`na` for links of K.knn and of S with no text evidence either way) |
| A6 | `exact` the tail's title occurs in a's text exactly as written (case-sensitive) / only case-folded / only by an alias form (the title without its trailing parenthetical or the text before its first comma; the last token) / not at all | exact / folded or alias / none |
| A7 | `ntok` the token count of c's title | 1 / 2 / 3 or more |
| A8 | `caps` the capitalisation of c's title | every word capitalised / mixed / lowercase |
| A9 | `lead` the first mention of c's title in a's text lies in a's first sentence | yes / no / `na` |
| A10 | `off` the character offset of that first mention over a's text length | up to 0.10 / 0.10 to 0.50 / above 0.50 / `na` |
| A11 | `cnt` the number of mentions of c's title in a's text | 1 / 2 / 3 or more / `na` |
| A12 | `qov` the number of content tokens of c's title that occur in the question (case-folded; a fixed English stop-word list in the code, no learning): question-conditioned, label-free | 0 / 1 / 2 or more |
| A13 | `rel` (KB only, S links) the relation id | **diagnosis only**: dataset-specific vocabulary, never in the shared program |

Attributes the lead listed that are **not** declared as such: "whether a two-hop closure exists" is declared as A4 (agreement with the question's ball) rather than common-neighbour existence, because common neighbours scale with the endpoints' degrees and the hub line is closed; "the reverse link" is A5; "the tail's type-free name overlap with the question" is A12; "direction" is A5; "mention-string shape" is A6, A7, A8; "position" is A9, A10.

**Cut points that are data-derived** (A2 only) are fixed once, on the seven quarters pooled with equal dataset weight, and are the same for every dataset and for R. Everything else is declared above.

## 6. The constrained program (one shared keep-rule)

**Keys.** A key is a cell of the joint partition of `grp` x (three attributes chosen as below) x their bins; at most 3 x 4 x 4 x 4 = 192 keys. A key absent from a dataset has zero columns there. **The attribute choice is a rule, fixed now:** on the quarter, per attribute A2 to A12, the stratified AUC of needed against control (within each (question, source) stratum that holds both, strata weighted by their needed weight) per judged cell, with the 95 percent bootstrap interval over questions (zap4's `boot`, as UL2P). The attribute's **score** is the minimum over the judged cells of |AUC - 0.5|, provided the sign of AUC - 0.5 is the same in every cell with an interval excluding 0.5; **SIGN_SPLIT** attributes (a sign that differs across cells) are excluded from the shared space and reported as dataset-specific patterns. The three attributes with the highest scores form the call's space K3 (A1 is always in). The four highest form K4 (beside). The baseline space K0 = `grp` x A2 (UL4's K1 on this population: the label and dot rule the page is compared with). Each attribute's **direction** delta (keep the larger or the smaller bins first) is the sign of its pooled AUC - 0.5 on the quarter, and is fixed before any solve.

**Variables.** x_k in {0, 1} for each key k (the rule keeps a link iff its key is kept), the **same x for every judged cell** (that is the universality constraint: no parameter carries a dataset index). tau in [0, 1].

**Columns (from the quarter).** For judged cell j and key k: r_jk = the weight of the needed units in k over the cell's total needed weight (gold-normalised reach); m_jk = the number of distinct needed links in k, e_jk = the weighted number of controls in k; price_j(x) = sum_k (m_jk + e_jk) x_k / sum_k (m_jk + e_jk); reach_j(x) = sum_k r_jk x_k.

**Objective and constraints.**
- minimise tau
- C1 (reach): reach_j(x) >= rho for every judged cell j; rho = 0.95 is the call (the user's 95+), 0.90 and 0.99 beside
- C2 (price): price_j(x) <= tau for every judged cell j
- C3 (S never removed, A.1): x_k = 1 for every key with `grp` = S; S free is solved beside
- C4 (monotone): for each of the three chosen attributes, x_k <= x_k' when k' is k with that attribute moved one bin in the direction delta_i and the other coordinates equal (the `na` bin is outside the order). The call carries this constraint (a rule that keeps a link and drops a better-looking one is not a pattern); it is dropped beside
- C5 (size, beside): sum_k x_k <= B for B in {12, 24, 48}
- C6 (universality): x has no cell index

**Solvers, and which is exact.**
- **(L) the relaxation** (x in [0, 1]; `scipy.optimize.linprog`, HiGHS): exact for the relaxation, a **lower bound** on every integer rule's tau; its duals say which cells' reach constraints bind, and tau*(rho) at 0.94, 0.95, 0.96 gives the price of a point of reach (sensitivity, UL4's form).
- **(U) the integer program** without C4 and **(M) with C4** (`scipy.optimize.milp`, HiGHS branch-and-bound, relative gap 0, 600 s limit): **exact when the status is optimal** (recorded); if the limit is hit the record says NOT_PROVEN with the incumbent and the bound, and the call uses the incumbent, said so.
- **(T) exhaustive enumeration** of the threshold rules (a link is kept iff, for each of the three attributes, its bin is at or beyond a threshold t_i in the attribute's direction, one threshold triple per group R and K, S kept): exact on its space (at most 4^6 combinations), and the carried form if M and T agree.
- The greedy cell order UL3 used (highest precision first until rho of the covered slots) is **not exact** for this problem; it is reported as the old form beside.
- The chain tau_L <= tau_U <= tau_M <= tau_T must hold (G-G6); a violation stops the read.
- Own optima (one cell alone, private x) and the **cost of universality** (tau* of the shared program minus the largest own optimum), the **cost of A.1** (S protected minus S free) and the **frontier** (the best reach the worst cell can have at price caps 0.30 to 0.80) are read as UL4 read them.

## 7. The per-dataset diagnosis (the user's (a); every dataset, quarter and rest)

For each judged cell: the gold distance mix (found / 1 / 2 / 3 / unreached), the needed units by step and by `grp` and label profile, **the needed share against the control share of each bin of each attribute** (the pattern, as a table, not a score), the stratified AUC of each attribute with its interval, the attributes' SIGN_SPLIT calls, and for the KB datasets the relations (A13) that carry the most needed weight with their control share. On the rest R the same tables are re-read with the committed bins and cut points (the pattern's stability, quarter against rest).

## 8. Gates (each failure stops that dataset's reading; none of its numbers is read)

- **G-G1** seeds: every question's union is 5 to 10 distinct rows (UL2H's `check_union`), and the carve is the declared one (ids and golds equal the carve file; WebQSP's dev disjoint from s1eval).
- **G-G2** distances: for every question of the slice, my gold distances equal the reach counts of `zul2h.Walker.walk` at 0, 1, 2 and 3 hops (array equality; the dev carve has no filed ledger to equal).
- **G-G3** needed units: on the selftest's random graphs the DAG equals brute-force path enumeration (a deliberately wrong DAG that keeps non-shortest paths is caught); on the real slice every reachable missed gold has at least one complete seed-to-gold path inside its units (array check), and the unit weights sum to the number of reachable missed golds.
- **G-G4** controls: the weights of a source's sampled controls sum to its exact frontier control count; no control is a needed unit of the same question.
- **G-G5** text: on 2,000 sampled R.mention links per dataset the offset and the count of A10 and A11 equal a regular-expression recount; the share of R.mention links with `exact` = none is printed (a mention link whose title does not occur is an alias or a sibling link, and is binned as such).
- **G-G6** solvers: on toy instances (8 keys, 3 cells) the integer program equals all-subsets brute force with and without C4 and with S protected; the threshold enumeration equals the integer program restricted to its space; the chain of section 6 holds on the real columns; a deliberately wrong objective is caught.
- **G-G7** no eval question is read (the code asserts that the carve it opens is the dev carve), and nothing is written outside `outputs/gp1/`.
- **Self-test** (`zgp1.py selftest`, laptop) runs G-G3, G-G5 (on a toy text) and G-G6 before any row is sent.

## 9. The calls, and what I expect (stated before any number; a failed one is reported, not explained away)

**Calls.**
- **Per attribute** (on the quarter and again on R): **SEPARATES** iff in every judged cell the stratified AUC's lower end is above 0.5, with the same sign, and the minimum |AUC - 0.5| over cells is at least 0.10; **WEAK** iff the mean over cells is at least 0.05 from 0.5 but the above fails; else **NO_SIGNAL**; **SIGN_SPLIT** as in section 6.
- **The rule (the call).** The M-program (S protected, rho 0.95) solved on the quarter is graded on R, per judged cell, by point values: SCREEN_IN iff reach retained (gold-normalised) >= 0.92, price retained <= 0.50 (UL3's thresholds, fixed before), **and** the **gold path retention** (the share of reachable missed golds that keep at least one complete shortest path with every link kept; a lower bound, since a longer path may survive a cut) >= 0.90. **UNIVERSAL_RULE** iff every judged cell of every read dataset is SCREEN_IN; otherwise **NOT_UNIVERSAL**, with tau* (dev), the binding cells, the cost of A.1 and of universality, and the improvement over the baseline K0 solved the same way on the same units (paired by the same questions: tau of K0 minus tau of K3, and the same two on R).
- **Rule-size read:** the number of keys the M-rule keeps, and whether the T form (a handful of thresholds) matches it within 0.02 in tau.

**Expectations.**
- **X1:** at least one attribute outside `grp`, `dot` and `votes` (the text shape and position attributes A5 to A11, or A3) is SEPARATES on the quarter. I put it near even odds; U2K's weights (tokens up, in-degree down) say shape matters for editor-chosen links, not that it matters for gold paths.
- **X2:** K3's tau* (S protected, rho 0.95, M-program, quarter) is at most 0.80, against K0's 0.85 or more on the same units. If K3 does not beat K0 by 10 points of tau the new attributes add little.
- **X3:** MuSiQue (R and K only, nearly flat cells in UL3) is the binding cell, or one of the two binding cells, again.
- **X4:** the passage datasets (MuSiQue, HotpotQA, 2Wiki, MultiHop-RAG, SQuAD) separate on text attributes, the KB datasets (MetaQA, WebQSP) do not, since a KB node's text is its name; their needed links are S links and relation ids, the dataset-specific diagnosis.
- **X5:** `votes` correlates above 0.70 with the tail's neighbour count on at least one dataset, and is dropped.
- **X6 (the one that matters):** **no universal rule reaches price 0.50 on every judged cell**: NOT_UNIVERSAL, with an improvement over K0 of 10 to 25 points of tau and the three datasets that bind named. I put the odds of UNIVERSAL_RULE at about one in four; a pass would be reported as a surprise.

## 10. What follows (fixed now)

- **UNIVERSAL_RULE on R:** declare GP2 (its own page): the one eval grade of the committed rule, and the path-level guard (the kept graph re-walked from the seeds over every question: pool size and ALL on the full carve), subset first. Nothing carries before that, and **PL8 is declared only on the user's word** that the link set is final.
- **NOT_UNIVERSAL:** report the tau, the binding cells and the per-dataset patterns; name, not declare, (1) the attribute as a **walk-time ordering** (rank the frontier by the pattern's score and cap by budget, the pool side, PL5 to PL7's lever, not a link filter), (2) a depth signal (UL4 10.6), (3) dataset-specific rules, which the universal ruling excludes.
- **No attribute separates:** the links gold paths need are not told apart from their neighbours by anything extractable here; this closes the list of section 5, not the line.
- The two other threads of this turn (UL5, the depth signal; SEM2 R8) are theirs; GP1 reads neither.

## 11. Code, rows, outputs

`outputs/mp_unified/zgp1.py` (new, mine; imports `zul2h` (Comp, Walker, Pop and the seed unions), `zul2p`, `zul3a` (`pair_dot`), `zul4lp` (solver pieces if reusable), `zap4`, `u1b`/`zmhb` (node text); edits none of them). Commands: `selftest`; `extract --ds D --part q|r [--host]` (units, controls, attributes; checkpointed per (version, part); writes `outputs/gp1/<d>/units_<part>.json` of aggregated columns only, no arrays); `screen` (laptop: AUCs, tables, attribute choice); `solve` (laptop: the programs on the quarter's columns; writes the committed rule record); `grade` (laptop: the rule on R's columns); `report`. Host rows are CPU, one per dataset and part, class names `gp1x-<d>-q` and `gp1x-<d>-r`, memory sized from UL2H's recorded peaks per dataset (`outputs/ul2p/hops/<d>/ledger.json`), not guessed; row inputs are files only; the mpr-cpu environment has no numba, so the rows run through `pylib_run.py`. The part-r rows are sent only after the quarter's rule record is committed. Outputs only under `outputs/gp1/`.

## 12. Disclosed

- The attribute list, the bins, the thresholds (0.92, 0.50, 0.90), rho, the caps (200 per source, 20,000 per question, 5,000 per gold), the quarter and the choice rule were fixed before any GP1 number, from reading UL3, UL4, U2J, U2K and UL2H's tables (section 1). The seeds on dev are the package's frozen lists (section 3).
- The pooled terciles of `dot` are the one data-derived cut; they are fixed on the quarter and not refit on R.
- The needed links are defined from the shortest paths in `any`: a longer path may serve a gold whose shortest path is cut, so gold path retention is a lower bound, and a rule that drops a needed link is not proved harmful. The price is at the sources of needed links, an estimate from a control sample, not the pool size.
- `votes` depends on the question's ball, so its value is not a property of the link alone; it is kept only if it is not a degree proxy (section 5).
- The statement of X1 to X6 was written after reading UL3 to UL4 and UL2H's tables. X6 is a belief.

## 13. Amendments made with the code, before any recorded run (26 Nov)

Written with `zgp1.py` (engine, extract and self-test). The toy self-test passed (mention statistics; the shortest-path links equal brute-force path enumeration on 120 random graphs and a longer-path DAG is caught; the levels equal `zul2h`'s walker; weights; the control gates). One timing run on the laptop took the first 8 questions of MultiHop-RAG's quarter; I read its gold-distance counts (18 of 21 golds already in the seeds, 3 at distance 1) and its sizes (16 units, 1,510 controls), and **no attribute value**. Nothing below uses a pattern number.

1. **Control cap.** Section 4's "200 per source, 20,000 per question" is one rule: a source keeps min(200, max(1, floor(20,000 / the number of sources of that question))) sampled controls (fixed hash order of the link key, weights as before, gate G-G4).
2. **Units.** A link serving two steps of one question is two units (the key is the link and its step). The weights are as section 4; the gate checks that they sum to the number of golds that have a unit.
3. **Rank** of a link = 1 + the number of its source's frontier links whose dot is strictly larger (ties share the rank); the dot is UL3's float32 dot of the two stored rows, computed for the **whole** frontier of every source of a needed unit (the rank needs it), in blocks of 16,384 pairs.
4. **Text codes.** `dir`: 0 none / 1 out (a's text names c) / 2 in / 3 mutual; `exact`: 3 as written / 2 case-folded / 1 alias / 0 none; `lead` 2, `off` nan and `cnt` 0 where a's text does not name c at all (the `na` bin). On a KB node the text is the entity's name, so these attributes read "does a's name contain c's name"; X4 expected no signal there and the code does not special-case it.
5. **G-G5** recounts up to 1,000 `R.mention` links per chunk of 50 questions (a regular expression with the same word boundaries, count and first offset), not 2,000 per dataset in total.
6. **Files.** Link-level working files (the units, the controls, every attribute, the gold-to-unit incidences) are written in chunks of 50 questions under `outputs/gp1/<d>/work/` and are **not committed**; they are stamped with `zgp1.py`'s sha256 and a chunk written by another version is recomputed. What is committed is the aggregated records. A run with `--limit` writes under `work_limit/` and `extract_<part>_limit.json` and is never read for a call.
7. **Host rows.** Names are `gp1s-<d>` (smoke, `--limit 5`, timing and memory only, no number read), `gp1xq-<d>` (part q) and `gp1xr-<d>` (part r), so that no two rows share the feeder's class (the first two name tokens); each row depends on `ul2c-grade-<d>`, runs `retry_cmd.py -- python outputs/mp_unified/zgp1.py extract ...` (no numba is needed: `zul2h` and `zul3a` import none), one CPU, memory from UL2H's recorded peak plus margin, then the measured peak of the smoke. `pop_dev.npz` of each dataset is a row input, so G-G1's carve check runs on the host too.
8. **Dev size.** The dev carve holds 1,534 (MuSiQue), 1,497 (MetaQA), 1,507 (WebQSP), 1,508 (HotpotQA), 1,496 (2Wiki), 1,498 (SQuAD) questions with a gold, and 224 for MultiHop-RAG's dev split; the quarter Q is 374 to 384 questions per dataset (224 questions, 56 in Q, for MultiHop-RAG). The dev carve has no UL2H ledger to equal, so G-G2 is the walker equality on every question read.
