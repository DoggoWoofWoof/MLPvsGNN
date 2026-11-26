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

## 14. Amendments made with the program code and the smoke rows, before any recorded quarter run (26 Nov)

Written with `zgp1p.py` (screen, solve, grade, report) and the amended `zgp1.py` (shards, relation vocabulary). The six smoke rows (`gp1s-<d>`, the first 5 questions of each quarter, one row per dataset) ran on the host; **I read their timing, memory, gate counts, gold-distance counts and the sizes of the units and controls, and no attribute value or pattern**. The program was run end to end on those 5-question files (`--smoke`, files `*_limit.json`, nothing committed from them). With 5 questions per dataset its printed tables carry no information and none enters a choice below.

1. **Judged cells: nine, not ten.** Section 1's list names nine cells (MultiHop-RAG RK; MuSiQue RK; SQuAD RK; MetaQA SRK; WebQSP SRK; HotpotQA RK and SRK; 2Wiki RK and SRK); its "ten" was a miscount. Nine are judged, seven datasets.
2. **Smoke cost and memory (host rows, one CPU).** SQuAD 4.5 s, 0.61 GB; MuSiQue 13 s, 0.45 GB; MetaQA 8 s, 1.64 GB; HotpotQA 364 s (210 s of it the dev slice), 9.8 GB; 2Wiki 686 s, 14.9 GB; **WebQSP 396 s for 5 questions, 11.8 GB, about 33 s per question after a 130 s graph load**, so its quarter (about 377 questions) is 3 to 4 hours in one row. The q rows ask: SQuAD 4 GB, MuSiQue 4 GB, MetaQA 6 GB, HotpotQA 16 GB, 2Wiki 20 GB, WebQSP 20 GB each.
3. **Shards.** `zgp1.py extract --shard I/N` takes the 50-question chunks whose number is I mod N (a chunk is a pure function of its questions, so sharding changes no value) and writes `extract_<part>_s<I>of<N>.json`; the reader merges the N files (chunks ordered by question offset, counts added, the capped flag recomputed from the totals; every shard must be present or the dataset is not read). Checked: MultiHop-RAG's quarter as two shards gives the same counts as its one run (141 golds, 99 found, 452 units, 20,162 controls). **WebQSP's quarter runs as four shard rows** `gp1xq-ws0` to `gp1xq-ws3` (each its own feeder class), the other six datasets as one row each (`gp1xq-<d>`).
4. **Undefined cells.** An attribute whose stratified AUC is undefined in a judged cell (no stratum there holds both needed and control links among its defined bins, as the text attributes on cells where the source text does not name the tail) cannot be SEPARATES or WEAK there: universality asks it of every judged cell. Its score is 0 and its call NO_SIGNAL; the per-cell tables still print what exists.
5. **K3 and K4 take only attributes with a positive score** (not SIGN_SPLIT, not the dropped `votes`). If fewer than three qualify, the space has those only (down to `grp` alone), said in the record; no zero-score attribute is added to fill a place.
6. **AUC details.** Computed on the defined (non-`na`) bins; the weights of a stratum are its needed weight; per cell the 95 percent interval is zap4's `boot` over the quarter's questions, its default resamples and seed; the pooled AUC (sign and direction) is the sum of the cells' numerators over the sum of their weights, questions pooled.
7. **The threshold form T.** For each attribute the ordered bins are kept from the best end in its direction down to a threshold (a threshold of 0 keeps none, of all bins keeps all); the `na` bin of an attribute is kept iff the threshold keeps every ordered bin. One threshold triple per group R and K (and S only on the S-free runs). Every T rule is monotone (C4) and keeps S, so tau_M <= tau_T holds by construction.
8. **The M rule written to the record** is the integer program's second stage: after tau* is found, the program is solved again with tau at most tau* (1e-9 slack) minimising the number of keys kept, so a key the quarter never saw is dropped. The record keeps the 0/1 vector of keys the quarter saw; R reports the share of its needed weight that falls in keys the quarter never saw (a stability read, not a gate).
9. **Gold path retention** follows the section 9 definition link by link: a gold keeps a path iff its first-step links kept include one ending at a node from which a kept second-step link continues, and so on to the gold; a gold at distance 1 needs one kept first-step link.
10. **Relations (KB diagnosis, A13).** The relation vocabulary of MetaQA and WebQSP is saved in the extract record (`rel_vocab`); the diagnosis lists the relations with the most needed weight and their control weight, nothing else uses it.
11. **Parts and files.** The quarter's screen, solve and rule record are `outputs/gp1/screen_q.json` and `rule_q.json` (committed, aggregates only, with the sha256 of the extract files they read); the grade on the rest is `grade_r.json`; `report.md` prints the tables. The r rows (`gp1xr-<d>`, WebQSP in four shards) are queued only after `rule_q.json` is committed.

## 15. Results on the dev quarter (26 Nov; the rest R is not yet read)

All nine quarter host rows (WebQSP in four shards) and the MultiHop-RAG laptop run finished; **every gate held and no cell is NOT_READ** (WebQSP's capped golds are 34 of 2,006, 1.7 percent, under the 5 percent line). Code: `zgp1.py` 0a9c6bc0, `zgp1p.py` 062986dc (the tolerance edit of 5e5d007d included). Files: `outputs/gp1/screen_q.json`, `rule_q.json`, the nine `extract_q*.json` records (WebQSP's four shards), `report.md` (all tables). The link-level work files (30 MB) are not committed. Disclosed: section 14.11 said the screen and rule records embed the sha256 of the extract files they read; they do not (the extract records are committed beside them, so git holds the shas).

### 15.1 The per-dataset diagnosis (the user's (a)), quarter

| cell | questions | golds | found by first stage | at 1 hop | 2 hops | 3 hops | unreached | needed units (steps 1/2/3) | needed weight in S / R / K | control weight in S / R / K |
|---|---|---|---|---|---|---|---|---|---|---|
| MHR/RK | 56 | 141 | 99 (70%) | 41 | 1 | 0 | 0 | 452 (88/12/0%) | 0/99/1% | 0.0/99.5/0.5% |
| MuSiQue/RK | 384 | 1017 | 512 (50%) | 241 | 205 | 55 | 4 | 26889 (40/56/4%) | 0/94/6% | 0.0/99.9/0.1% |
| SQuAD/RK | 375 | 375 | 348 (93%) | 17 | 7 | 3 | 0 | 1327 (54/40/6%) | 0/96/4% | 0.0/99.8/0.2% |
| MetaQA/SRK | 375 | 3228 | 14 (0%) | 409 | 1769 | 1036 | 0 | 74373 (10/71/18%) | 51/46/3% | 39.1/58.7/2.3% |
| WebQSP/SRK | 377 | 2006 | 80 (4%) | 1192 | 704 | 30 | 0 | 193807 (25/74/1%) | 55/41/4% | 1.7/98.3/0.0% |
| HotpotQA/RK | 377 | 754 | 553 (73%) | 146 | 45 | 10 | 0 | 1405 (70/24/6%) | 0/96/4% | 0.0/100.0/0.0% |
| HotpotQA/SRK | 377 | 754 | 553 (73%) | 172 | 22 | 7 | 0 | 918 (68/24/8%) | 84/15/1% | 8.8/91.1/0.0% |
| 2Wiki/RK | 374 | 916 | 552 (60%) | 301 | 55 | 8 | 0 | 1744 (61/32/7%) | 0/98/2% | 0.0/100.0/0.0% |
| 2Wiki/SRK | 374 | 916 | 552 (60%) | 322 | 34 | 8 | 0 | 1540 (61/31/9%) | 77/21/2% | 3.3/96.7/0.0% |

Where the needed links sit, **needed share against control share** (the pooled table, not stratified; the stratified AUCs are in 15.2):

| cell | tail in the question's top 30 | dot high | exact title | dir one-way or mutual | votes 3+ |
|---|---|---|---|---|---|
| MHR/RK | 90% vs 36.9% | 98% vs 56% | 0% vs 0.0% | 57% vs 32% | 95% vs 69% |
| MuSiQue/RK | 56% vs 1.8% | 63% vs 14% | 43% vs 1.8% | 78% vs 99% | 41% vs 97% |
| SQuAD/RK | 54% vs 11.3% | 80% vs 38% | 27% vs 6.9% | 40% vs 84% | 56% vs 91% |
| MetaQA/SRK | 66% vs 10.5% | 72% vs 78% | 1% vs 0.0% | 2% vs 2% | 36% vs 51% |
| WebQSP/SRK | 35% vs 1.3% | 85% vs 28% | 19% vs 0.0% | 24% vs 4% | 29% vs 97% |
| HotpotQA/RK | 78% vs 0.1% | 79% vs 13% | 68% vs 0.0% | 96% vs 100% | 27% vs 19% |
| HotpotQA/SRK | 83% vs 0.2% | 85% vs 11% | 60% vs 0.1% | 93% vs 97% | 22% vs 24% |
| 2Wiki/RK | 82% vs 0.1% | 86% vs 7% | 82% vs 0.1% | 98% vs 100% | 15% vs 39% |
| 2Wiki/SRK | 83% vs 0.1% | 90% vs 7% | 78% vs 0.1% | 97% vs 99% | 12% vs 43% |

(The other attributes' tables, the label profiles and the relations of MetaQA and WebQSP are in `report.md` and `screen_q.json`.) Readings, as tables say them and no more: the needed links point at tails that the question's own dense ranking already places in its top 30 (54 to 90 percent of needed units in eight of nine cells against 0.1 to 11 percent of controls, MultiHop-RAG's 609-node graph 37; WebQSP is the exception with 35 against 1.3), and at tails with a high pair dot (63 to 98 percent against 7 to 56 percent) in every cell except MetaQA (72 against 78). In HotpotQA, 2Wiki and MuSiQue 43 to 82 percent of the needed units have an exact title match (A6) against 0.0 to 1.8 percent of controls; in MetaQA it is 1 percent, in WebQSP 19. **MetaQA is the cell that behaves differently from the rest**: the first stage finds 14 of its 3,228 golds, 87 percent of the missed ones are 2 or 3 hops out, half of its needed weight sits on S links (51 percent) and the rest on typed relations, and its dot, dir, exact and caps AUCs lie below 0.5 where the other cells' lie above.

### 15.2 The attribute calls (stratified AUC [95% interval], per judged cell)

| attribute | call | score | direction | pooled AUC | MHR | MuSiQue | SQuAD | MetaQA | WebQSP | HotpotQA RK | HotpotQA SRK | 2Wiki RK | 2Wiki SRK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 dot | SIGN_SPLIT | 0 | +1 | 0.571 | 0.694 | 0.540 | 0.548 [0.491,0.593] | 0.484 | 0.700 | 0.556 | 0.580 | 0.623 | 0.640 |
| A3 rank | SIGN_SPLIT | 0 | +1 | 0.520 | 0.237 | 0.475 | 0.383 | 0.581 | 0.491 [0.471,0.508] | 0.479 [0.452,0.504] | 0.458 | 0.413 | 0.406 |
| A4 votes | **WEAK** | **0.026** | +1 | 0.576 | 0.646 | 0.565 | 0.581 [0.510,0.649] | 0.567 | 0.612 | 0.573 | 0.545 | 0.535 | 0.526 |
| A5 dir | SIGN_SPLIT | 0 | +1 | 0.518 | 0.639 | 0.557 | 0.497 [0.443,0.553] | 0.453 | 0.514 [0.492,0.540] | 0.637 | 0.637 | 0.722 | 0.720 |
| A6 exact | SIGN_SPLIT | 0 | +1 | 0.549 | 0.585 | 0.595 | 0.521 [0.467,0.577] | 0.462 | 0.576 | 0.704 | 0.675 | 0.753 | 0.748 |
| A7 ntok | WEAK | 0 (0.500 in MHR and SQuAD) | -1 | 0.365 | 0.500 | 0.460 | 0.500 | 0.358 | 0.267 | 0.479 [0.447,0.513] | 0.513 | 0.499 | 0.519 |
| A8 caps | SIGN_SPLIT | 0 | +1 | 0.577 | 0.485 | 0.507 | 0.503 | 0.477 | 0.771 | 0.544 | 0.533 | 0.580 | 0.585 |
| A9 lead | SIGN_SPLIT | 0 | +1 | 0.516 | 0.517 | 0.510 | 0.516 | 0.500 | 0.490 | 0.450 | 0.435 | 0.568 | 0.569 |
| A10 off | SIGN_SPLIT | 0 | -1 | 0.495 | 0.472 | 0.496 | 0.493 | 0.519 | 0.462 | 0.600 | 0.613 | 0.458 | 0.460 |
| A11 cnt | NO_SIGNAL | 0 | -1 | 0.479 | 0.504 | 0.481 | 0.509 | 0.500 | 0.494 | 0.464 | 0.453 | 0.477 | 0.477 |
| A12 qov | SIGN_SPLIT | 0 | -1 | 0.486 | 0.611 | 0.546 | 0.500 | 0.450 | 0.510 | 0.546 | 0.547 | 0.496 | 0.503 |

(Intervals are printed only where they contain 0.5; all are in `report.md`.) **No attribute SEPARATES** (none has the lower end above 0.5 with the same sign everywhere and a minimum distance of 0.10). Eight are SIGN_SPLIT, two WEAK (`votes`, `ntok`), one NO_SIGNAL. `votes` is the only attribute with one sign in all nine cells; its smallest distance from 0.5 is 0.026 (2Wiki SRK) and its guard against being a degree proxy held (largest Spearman with the tail's neighbour count 0.53 on WebQSP, under 0.70), so **K3 = K4 = grp x votes, nine keys**: the attribute choice rule left one attribute beside `grp`. `ntok` is 0.500 (undefined) in MultiHop-RAG and SQuAD, so its score is 0 by section 14.4.

### 15.3 The program on the quarter (S protected, rho 0.95, M with monotone keys)

- **tau\* = 0.99997** (K3): chain L 0.9515 <= U 0.99997 = M 0.99997 = T 0.99997, M OPTIMAL at gap 0. The rule keeps 8 of the 9 keys (it drops K|votes=1 only); the T form (R: all three vote bins, K: the two better bins) equals M, so the rule is two thresholds. Price on the quarter 0.993 to 1.000 in every cell; reach 0.957 (MuSiQue) to 0.9999.
- **K0 = grp x dot: tau\* = 1.000** (L 0.9661); all seven keys the quarter saw are kept. **Improvement of K3 over K0: 3.0e-5 of tau (0.003 points).** The greedy order of UL3 gives 1.000 on both.
- **Beside:** S free 0.99997 (**cost of A.1 = 0**); rho 0.90: 0.99993, rho 0.99: 1.000; size caps 12, 24, 48: 0.99997 (the cap does not bind); relaxation at rho 0.94, 0.95, 0.96: 0.9419, 0.9515, 0.9612, so **a point of reach costs about a point of price**: the best fractional keep-rule on these keys is no better than thinning every key alike (tau_L = rho + 0.0015); **cost of universality** (shared minus the largest own optimum) **4.0e-5**.
- **Own optima (private x, grp x votes):** MHR 0.832, HotpotQA SRK 0.526, MetaQA 0.977, and 0.998 to 1.000 for the other six cells. The two cells that can be tightened by their own votes keys are MHR and HotpotQA SRK (S carries 84 percent of that cell's needed weight).
- **Duals** (relaxation): the reach constraints that bind are MultiHop-RAG (0.930) and MuSiQue (0.039) for K3, MuSiQue (0.678) for K0. **Frontier** (the best reach the worst cell can have at a price cap): 0.30 infeasible with S protected, 0.40 reach 0.000, 0.50 to 0.80 reach 0.012.

### 15.4 Expectations of section 9, quarter part

- **X1 FAILED:** no attribute outside `grp`, `dot`, `votes` SEPARATES (none does, 0 of 11).
- **X2 FAILED:** K3's tau\* is 0.99997, not 0.80 or less; K0's 1.000 does meet "0.85 or more"; the 10-point improvement is 0.003 points.
- **X3 HELD, with a caveat:** MuSiQue is one of the two binding reach constraints for K3 (the other is MultiHop-RAG) and the binding one for K0; every cell's price is 0.99 or above, so "binding" says which reach limits the rule, not which price.
- **X4 FAILED as stated, on both halves:** among the passage datasets `exact` is 0.68 to 0.75 only in HotpotQA and 2Wiki (MuSiQue 0.595, MultiHop-RAG 0.585, SQuAD 0.521 with an interval over 0.5), and the text attributes of MultiHop-RAG and SQuAD are 0.5 or undefined; the KB datasets, which I expected not to separate on text, carry the largest distances in the table: WebQSP `caps` 0.771 and `ntok` 0.267, MetaQA `ntok` 0.358 (the needed links have shorter names), while `ntok` is 0.46 to 0.52 in every passage cell. The text patterns are dataset-specific, not passage-versus-KB.
- **X5 FAILED:** `votes` is not a degree proxy at the declared guard (0.04 to 0.53) and is kept.
- **X6 (the one that matters), the quarter part: HELD in direction and wrong in size.** No rule reaches price 0.50 (the program's own optimum prices 0.9999); the expected 10 to 25 points over K0 is 0.003 points. The call itself is graded on R.

### 15.5 What I state before any R number (the rest R is read once, with these cut points and this rule)

The quarter's rule keeps 99.3 to 100 percent of the candidate weight in every cell, so I expect:
1. **NOT_UNIVERSAL on R, with no cell SCREEN_IN**: the price retained is 0.95 or more in every cell of R (every cell fails the 0.50 line); the call changes only if the price is 0.50 or less in all nine cells, which I put under 1 percent.
2. Reach retained is 0.92 or more in at least seven of nine cells (MuSiQue 0.957 and SQuAD 0.963 on the quarter are the nearest to the line; SQuAD's quarter holds 27 reachable missed golds, so its R value moves most).
3. **Stability:** `votes` stays above 0.5 in all nine cells on R, and no attribute SEPARATES on R (about 85 percent); `dot`, `exact`, `dir` and `rank` keep the quarter's sign in at least seven of nine cells each.
These are written after the quarter, before any R value; they are not hypotheses the quarter tested.

### 15.6 What follows (section 10, NOT_UNIVERSAL branch; named, not declared)

The link attributes of section 5 do not tell the links gold paths need from their neighbours well enough, in one universal form, to take any weight off the walks: the price of 95 percent reach is about 100 percent. This closes the list of section 5, not the line. Named for the user to choose: (1) **the walk-time ordering**: the pooled tables above show the needed links pointing at tails that the question already ranks high (54 to 90 percent in the top 30 in eight of nine cells against 0.1 to 11 percent of controls, MultiHop-RAG's controls 37) and at high pair dot in every cell but MetaQA; that is the question-guided lever of PL5 to PL7 (rank the frontier by the question and cap it by budget), not a link filter, and MetaQA is where it is weakest; (2) a depth signal (UL4 10.6; UL5 32b3d4a5 found no label-free statistic of the five sets that sets one); (3) dataset-specific rules (HotpotQA and 2Wiki show a clear pattern in `exact` and `dir`, AUC 0.64 to 0.75; MHR and HotpotQA SRK can be tightened by their own votes keys), which the universal ruling excludes. GP2 is not declared; PL8 only on the user's word that the link set is final.
