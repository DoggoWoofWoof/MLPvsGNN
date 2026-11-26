# GP3 — the node-count rule: an ordered three-hop walk with per-hop distinct-node budgets, one exact program

Declared 26 Nov 2026, before any GP3 number exists. Code: `outputs/mp_unified/zgp3.py` (the walker and the extraction; frozen when the first row is queued) and `outputs/mp_unified/zgp3s.py` (the program, `solve`, `grade`), both committed with this page. They import `zgp1.py` (sha256 `0a9c6bc0026f71849fc4cacee333f9665390ce2bb3616ad9e338d7abe9df2c14`), `zgp1p.py` (`062986dcc0b6f5f5020c3094523228976f86839563dcf8e9d3fe851e80730839`) and `zgp2.py` (its pins are gated by `Z.gate_pins()`) read-only. Neither is edited. The PL, UL5 and SEM2 code is not imported.

## 1. Why this page exists

GP2 (`docs/GP2_UNIVERSAL_LINK_SCORE.md`, sections 11 to 17) ended on a question to the user: judge a link rule by its **price** (the share of candidate links kept) or by **how many nodes a question's pool holds**, and whether the 2,500 counts S. The user answered: **"stop waiting for my answer do everything you are asking me"**, which settles the question as the parent session put it: further link rules are judged by **distinct nodes per question, S counted as nodes**. The standing rulings that bind this page, in the user's words:

- **"coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how much ever is possible"**
- **"No universal one-hop rule exists. / we need to make it universal, tell the bounds or costs for that to happen i told you my limits, 95+ all coverage, as close as possible to 2.5k nodes"** (UL4's answer was no universal one-hop rule; the ask was the bounds and the costs of making one)
- **"we need to have universal rules"**: one rule, the same parameters, on every judged cell.
- **"well the issue with that is R has much much more links than S, we cant afford to have multiple useless links draining our walks and need to have much smarter walks too which can be done two ways eliminate useless links and use the question to guide"**.
- S is never removed (A.1); no hub, degree or IDF attribute; no learning; no LLM; no GPU; subset first, the rest of dev once; `s1eval` is never read.

**The design on this page is mine** (the user asked for the problem to be solved, not for this family): the family of section 4, its ladder, its keys, its program and the calls of section 7. Where the user's words and mine differ I name it.

What GP1 and GP2 left open, and why links are not nodes:

1. GP2's own optima were in **links per question**: MultiHop-RAG 163, HotpotQA RK 133, 2Wiki RK 119, SQuAD 1,862, MuSiQue 4,408, MetaQA 6,871, WebQSP **95,611** (K2, reach 0.952, 5.6 percent of its candidates). S alone exceeds 2,500 links on four cells. A link says nothing about how many **distinct** nodes it adds: many links point at the same tail, and many tails are already in the first stage.
2. GP2's best lever was a **per-question budget over a source-relative order** (K2, C2). GP3 turns that into a budget in nodes, per hop, over an **ordered walk**: the walk keeps the first N nodes of an order, so its cost is the number of nodes by construction.
3. UL5's oracle (golds read, a different universe) already says what the floor looks like: 541 (2Wiki), 1,831 (MuSiQue), 2,745 (MetaQA), 9,301 (WebQSP) mean nodes; its uniform level (the uncut pool) 39,572 on WebQSP. A rule with no label can only be at or above the oracle. **I expect WebQSP to need far more than 2,500 distinct nodes** (section 8); the page is built so that this is a finding with a number and not a failure of the rule.

## 2. What I have seen before this page (nothing of GP3)

- **GP1 and GP2 quarter results** (the numbers cited in section 1; GP1 838f199a, GP2 4e60e909 and 103d36cf), and UL5's section 11.2 table (its oracle and uniform nodes, which I quote only as context). None of these is a GP3 quantity.
- **No GP3 number.** The only runs of the new code are the self-tests on toy graphs (no real data): `zgp3.py selftest` and `zgp3s.py selftest`, both PASS (section 9, G3-3). The code was **never run on a real dataset** before this page; the `time` command (section 10) is the first.
- The rest of dev is unread by GP1, GP2 and GP3 (GP2's R grade, GP1 section 15.7, is pending in the same session). `s1eval` is unread.
- Code read before writing: `zgp1.py` (graph, levels, shortest-path neighbours, dev slice), `zgp2.py`, `zgp2a3.py` (the scores), `zul5st.py` and `zpl5g.py` (UL5's and PL's walks), `zul2h.py`, `zul3a.py` (`pair_dot`), `zpl2g.py`.

## 3. The measure

For a question q of a dev population, the **pool** is the set of distinct nodes

    Pool(q) = Base_b(q)  ∪  Seeds(q)  ∪  Kept_1(q)  ∪  Kept_2(q)  ∪  Kept_3(q)

- `Base_b` is the first stage as UL5's family fixes it: the first b nodes of the reciprocal-rank fusion (constant 60) of the dense and SPLADE top-1000 lists, **b = 25 for MultiHop-RAG, HotpotQA and MetaQA, 15 for SQuAD, 400 for 2Wiki, MuSiQue and WebQSP**. `Seeds` are the dense and SPLADE top fives (the walk's start; the gate G3-1 checks it).
- **The count is |Pool(q)|, distinct nodes, base and seeds included, S counted** (UL5's "nodes" also include the base: SQuAD's 15 is its base at level 0). The cell's number is the mean over its questions.
- **ALL** is the share of the cell's questions whose pool holds **every** gold. **Coverage target 0.95.**
- A **cell** is a dataset and a layout, as in GP1 and GP2: MultiHop-RAG RK, MuSiQue RK, SQuAD RK, MetaQA SRK, WebQSP SRK, HotpotQA RK and SRK, 2Wiki RK and SRK: nine cells. S is a part of the link set only in the SRK cells.
- **Populations (subset first):** the dev quarter is dev index mod 4 = 0; the rest of dev is the other three quarters. GP3 reads the quarter, chooses, files; the rest of dev is read **once** (section 6). Dev sizes: MultiHop-RAG 224, SQuAD 1,498, HotpotQA 1,508, 2Wiki 1,496, MuSiQue 1,534, MetaQA 1,497, WebQSP 1,507.

## 4. The declared family: an ordered three-hop walk with per-hop distinct-node budgets

**The graph** is GP1's any-link undirected graph of the cell (the cell's link components: S in the SRK cells, W, T0, K in all), with BFS **levels** from the seeds: level 0 the seeds, 1 and 2 their one- and two-hop balls, 3 everything else. A gold's **distance** is 0 to 2 from its level, 3 if it has a neighbour at level 2, else 4 (never reached by three hops).

**The walk** keeps, at each hop h = 1, 2, 3, from the nodes kept at hop h-1 (the seeds at h = 1):

1. the **candidates**: nodes at level h adjacent to a kept parent (a **shortest-path-forward walk**: a node is a hop-h candidate only at its own level, so a gold at distance d can enter at hop d only);
2. the **S-tails**, the candidates that have at least one S-labelled incoming link from a kept parent: **S whole**, always kept, **free** of the budget but **counted** in the pool (S is never removed, A.1);
3. of the other candidates, the **first N_h in the order of a score** (section 5); ties go to the lower node id.

The pool counts a kept node once, however many hops or links name it, and not at all if it is already in `Base ∪ Seeds`. **The budgets are absolute node counts per hop, not a share and not a link count**, taken from one ladder.

**The ladder:** N ∈ {0, 10, 30, 100, 300, 1,000, 3,000, 10,000, 30,000} (nine steps), and the rule is a **triple** N_1 ≥ N_2 ≥ N_3 from it: 165 triples. N = 0 keeps S-tails only (in an RK cell, nothing: the pool is `Base ∪ Seeds`; gate G3-4). 30,000 per hop is the ladder's top; where a hop has fewer candidates the budget does not bind. Non-increasing budgets are the declared shape (the walk narrows with depth; PL's own windows do the same); I do not test an increasing triple.

**What the walker stores** (so every triple is graded exactly, offline): per question, per score, per ladder step, the pool sizes and the **place of every gold** in the order of its hop (0 = an S-tail, k = the k-th other node, none if it is not a candidate). A walk with the budgets (N_1, N_2, N_3) keeps exactly the nodes that a prefix of each order names, because the order at hop h depends on the parents kept at hop h-1 but not on N_h: the walker computes the hop-2 order once per N_1 and the hop-3 order once per (N_1, N_2). **The golds enter only as probes whose place in an already fixed order is read** (gate G3-5 runs the walk with and without them and compares every size).

**Differences from the PL and UL5 walks (disclosed; their numbers are context, not comparable cell for cell).** PL's walk is a visited-set walk with a 512-entry per-row window, question-cosine keys (vr, fu, rr) and a budget split `ceil(rem / (H - h + 1))`; a gold at distance 1 that misses the hop-1 cut can still be found at hop 2 through a link between two level-1 nodes. GP3's BFS-DAG walk does **not** do that: the gold must enter at its own level. So GP3 can only lose coverage against the PL walk at equal kept nodes, never gain; the **uncut ball (every candidate kept at every hop) is the same pool in both**, which gives a context check (expectation X9). There is no window and no per-row cut in GP3: the budget is on distinct nodes of the hop.

## 5. The scores (the orders of the non-S candidates)

For a candidate c at hop h, over the **kept parents' incoming links** to c: `u` = GP2's first-stage rank bin of c (7 bins: the smaller of its ranks in the dense and SPLADE top-1000 lists, `Z.rank_bin`; a node in neither is bin 0); `rk` = the smallest **source rank** over the incoming links (a link's source rank is 1 + the number of links of the same source with a strictly larger pair dot, GP1's A3 `rank`, computed among the source's level-h links); `vt` = the number of kept parents that name c (clipped at 255); `dt` = the largest pair dot over the incoming links (the pair dot of the two stored rows, `pair_dot`). Keys (larger earlier; the composite integer key equals the tuple order, self-tested):

| score | order | what it is |
|---|---|---|
| K1 | (u, dt) | first-stage rank bin, then the dot (GP2's `ur` then `dot`) |
| K2 | (-rk, dt) | the **question-independent source rank**, then the dot (the score GP2 found to hold WebQSP at reach 0.952) |
| K3 | (u, -rk, vt, dt) | GP2's K3 (ur, rank, votes, dot), with `vt` = kept parents (see below) |
| K4 | (dt) | the dot alone |
| R0, R1 | a hash of (question, node) | **two random orders: the null** (they use no attribute) |

**K5 and K6 (GP2's text-window scores) are not run:** text windows each moved GP2's price by less than 0.0004 (GP2 section 15, phase 5), and they need the question's text columns on the host. They are named, not declared. **`vt` differs from GP2's votes:** GP2 counts all level-(s-1) neighbours of the tail; GP3 counts the **kept** parents, which is what a walk can know. `rk` is clipped at 524,287 and a dot at float32 (the self-test checks the composite key against the tuples at those clips).

## 6. The program and the order of work

For every score K and triple t, the quarter gives, per cell, ALL(K, t) and the mean pool size Nodes(K, t). **Exact enumeration** over the 165 triples (coverage is not monotone in N_1 or N_2: a larger N_1 admits better and more parents but dilutes the hop-2 order, so no search that assumes monotonicity is used).

- **P1 (the user's "95+ at the smallest node count"):** minimise the **worst cell's mean nodes** subject to ALL ≥ 0.95 **in every cell**, one triple for all cells; ties by the smaller sum of cells' nodes, then the earlier triple. If a cell cannot reach 0.95 even at the ladder top, P1 is taken **over the cells that can**, and the cells left out are named in the filing (this is the honest reading of "95+ everywhere" when a cell's ceiling is below it; the left-out cells enter P2).
- **P2 ("as close as possible to 2.5k"):** for each ceiling N ∈ {2,500, 3,500, 5,000} on the **worst cell's mean nodes**, maximise the **worst cell's ALL**. This is the bound-and-cost table the user asked for.
- **Own optimum** per cell and score: the fewest mean nodes at ALL ≥ 0.95 in that cell alone (the price of universality is the universal triple's nodes over this); the **floor** (N = 0) and the **ladder top** per cell.
- **The null:** the same programs on R0 and R1.
- **The final rule:** the simplest K among K1 to K4 (order K4, K2, K1, K3) whose P1 worst-cell nodes are within 5 percent of the least; with no K feasible, the K with the largest P2 min ALL at the 5,000 ceiling.
- **Order of work:** (1) this page and the code, committed; (2) the label-free `time` command, then an amendment (section 11) if only a cost knob changes; (3) the quarter extraction on the host (laptop for MultiHop-RAG), `zgp3s.py solve`, the quarter filed on this page with the expectations scored; (4) the rest of dev, once, `zgp3s.py grade` (the quarter's final rule and per-cell triples are applied; **nothing is re-chosen**); (5) filed, per-dataset nodes at ALL 0.95 reported.

## 7. Calls (fixed before any number)

- **C1, ordering matters:** K beats the null if its P1 worst-cell nodes are ≤ 0.9 × the smaller of the two random orders' (or the random orders are infeasible and K is not). `BEATS_NULL`, `NOT_BEATING_NULL` or `NO_P1` is filed for the final score; the per-cell own optima against the random ones are filed beside.
- **C2, the 2,500:** the **universal rule MEETS 2,500** if the final rule's P1 worst-cell mean nodes ≤ 2,500 (equivalently P2 at 2,500 has min ALL ≥ 0.95); otherwise the filing names the worst cell, the factor over 2,500, and the cells that are inside 2,500 at their own optimum and at the universal triple.
- **C3, per-dataset nodes at ALL 0.95** (the report the parent asked for): per cell, the nodes at the universal triple and at the cell's own optimum, with the UL5 oracle and uniform beside as context.
- **C4, the rest of dev HELD** if, at the quarter's final rule, every cell's ALL on the rest of dev is ≥ **0.92** (point value; Wilson 95 percent intervals filed beside) and no cell is CAPPED (section 9, G3-6); else `NOT_HELD`. The rest of dev is read **once**; `grade` refuses a second run (G3-8).
- **C5, one rule:** the filed rule is **one score and one triple on all nine cells**. A per-cell optimum is a bound, not a rule.
- **C6, the link set:** whatever the result, **the link set stays base and PL8 stays undeclared**. A GP3 rule changes the walk's order and budgets; adopting it is the user's call, and carrying it to a layout is a separate declaration.

## 8. What I expect (written before any GP3 number)

- **X1, the order matters.** The final rule beats the null (C1) with a ratio ≤ 0.5 on the worst cell. Reason: GP1 section 15.8, the needed links point at tails in the top 30 of their source's frontier by pair dot in 54 to 90 percent of cases against 0.1 to 11 percent of controls; a random order spends the budget on tails nobody needs.
- **X2, WebQSP decides P1 and is far above 2,500.** The worst cell is WebQSP SRK; the final rule's P1 worst-cell mean nodes are **between 10,000 and 30,000** (point guess 18,000). UL5's oracle (golds read, per-question) is 9,301 and its uniform level 39,572; a label-free order cannot reach the oracle. **The 2,500 is not met by the universal rule (C2: not met, probability about 0.9).**
- **X3, own optima.** Five or six of the nine cells have an own optimum ≤ 2,500 nodes (MultiHop-RAG, SQuAD, HotpotQA RK and SRK, 2Wiki RK and SRK); **MuSiQue 3,000 to 12,000, MetaQA 3,500 to 6,000** (its S counted whole), **WebQSP 10,000 to 30,000**. The floor (N = 0) alone is above 2,500 on WebQSP and MetaQA only if S whole is: S alone exceeded 2,500 **links** on four cells, but its **distinct tails** are fewer; I expect the floor to be below 2,500 on all cells but WebQSP.
- **X4, the price of universality.** The universal triple costs at least twice a cell's own optimum on at least four cells (the triple is set by WebQSP; the cells with small frontiers pay for it only where the walk has candidates to spend on, so the cost is in cells with large hop-2 and hop-3 balls: MuSiQue, MetaQA, HotpotQA, 2Wiki).
- **X5, the scores.** K1, K2 and K3 are within 30 percent of each other on P1 worst-cell nodes (no score separates from the others by more than the null does); K4 (the dot alone) is the weakest of the four in at least five of the nine cells' own optima.
- **X6, P2 at 2,500.** The worst cell's ALL at the best triple under a 2,500 ceiling is **between 0.35 and 0.75** (WebQSP). At 5,000 it is below 0.95.
- **X7, the rest of dev holds** (C4): every cell's ALL ≥ 0.92 at the quarter's final rule. The selection is among 165 triples on about 375 questions per cell (56 on MultiHop-RAG), so I expect the point values to fall 0.00 to 0.03 under the quarter's; MultiHop-RAG's interval is wide.
- **X8, the ladder's top reaches 0.95 on every cell except possibly WebQSP** (UL5's uncut ALL there is 0.956, the closest to the line); if WebQSP's ladder-top ALL is below 0.95, P1 is taken over the other eight cells, WebQSP's ceiling is a finding, and the final rule is read as such.
- **X9, the context check.** The ladder top's ALL equals UL5's uncut ALL (MultiHop-RAG 0.996, SQuAD 0.952, HotpotQA 0.968, 2Wiki 0.955, MuSiQue 0.963, MetaQA 0.961, WebQSP 0.956) to within 0.02 on each cell, because both pools are the three-hop ball (the datasets' layouts and populations differ slightly: the UL5 numbers are context, a gap above 0.03 is **reported as a finding about the walk's universe**, not hidden).

## 9. Gates

- **G3-1 (data identity):** the dev lists rebuilt equal the slice's count and the first 50 questions' seeds equal the first-stage top fives; the UL2P data gates of GP1 pass; `extract_<part>*.json` records `status read`, every shard of the part present.
- **G3-2 (pins):** `zgp1`, `zgp1p`, `zgp2` and what they pin match their sha256s.
- **G3-3 (self-tests, before any real row):** `zgp3.py selftest` (the composite keys equal the tuple orders; `src_rank`; 70 toy graphs × 6 scores × 20 triples = **8,400 pools and 24,600 gold checks equal a brute-force walk**; the sizes do not depend on the probes) **PASS**; `zgp3s.py selftest` (the exact grading of a chunk equals the naive pools of the same 8,400 walks; P1 and P2 equal a plain search on 400 random matrices, with and without a cell subset; a planted case catches a wrong objective; the final-rule tie order) **PASS**. Both were run on this code before this page.
- **G3-4 (S whole and nothing else at N = 0):** in an RK cell the walk keeps no node at N = (0, 0, 0), checked on every question; in an SRK cell the floor is the S-tails of the seeds' links.
- **G3-5 (no label in the walk):** the first two questions of every chunk are re-run without probes and every size is compared.
- **G3-6 (the gather cap):** a hop whose gathered neighbour entries exceed **12,000,000** has its parents thinned in hash order and is flagged; a cell with more than **2 percent** of its questions flagged is **CAPPED**: its numbers are filed and its call is not made until an amendment (a cost knob only) has decided it.
- **G3-7 (versions):** the chunks of a whole part carry **one** zgp3 sha256, which `solve` and `grade` record; the chunks' question sets equal the part.
- **G3-8 (the rest of dev is read once):** `grade` refuses to run if `outputs/gp3/rule_r.json` exists.

## 10. Cost and operations

- The walk's cost is dominated by the pair dots over the hop-2 and hop-3 link tables (WebQSP: millions of links per question in GP2's count; each a 1,152-wide row dot). The label-free `python outputs/mp_unified/zgp3.py time --ds D --n 20 [--host]` runs the first 20 quarter questions with no gold, prints seconds, link counts, candidate counts and the capped flag. **The cost knobs (the gather cap, the chunk size, the shard count) may be changed by an amendment decided from `time` only**; the ladder, the keys, the programs and the calls may not.
- **Host rows**, one per dataset (WebQSP four shards), `--threads 1`, `--host`, **no GPU**, one class name per row, sized from measured peaks (GP1's 20 GB rows are the reference), queued **behind the current link rows only if memory is free**, with the inputs of GP1's rows plus `zgp3.py`, `zgp3s.py`, `zgp2.py`, `zgp2a3.py`, `zgp1.py`, `zgp1p.py`. The laptop runs MultiHop-RAG only.
- **Checkpointing:** chunks of 20 questions are written as they finish (`outputs/gp3/<d>/work/<cell>_<part>_<q0>.npz`, never committed; the aggregates, `rule_q.json`, `rule_r.json` and the reports, are, with `git add -f`); a rerun resumes by chunk and sha. Nothing here throttles or contends for the GPU, which is not ours.

## 11. Disclosures, and what this page does not do

- **The walk universe differs from PL's and UL5's** (section 4): a gold must enter at its own level, there is no per-row window, and the budgets are distinct-node counts per hop. GP3's numbers are therefore comparable with UL5's oracle and uniform only as context, and the ladder-top check of X9 is how I will say how far the universes differ.
- **`vt` is kept parents** (section 5); GP2's votes were all level-(s-1) neighbours. K5 and K6 are not run.
- **The base is UL5's family b per dataset** and is counted (so the floor of a cell is at least its base: 15 to 400 nodes).
- **Selection among 165 triples on a quarter** is optimistic by construction; C4 is the held-out check; the rest-of-dev report also files, **as a descriptive column and not a call**, the nodes re-optimised on the rest.
- **Not done:** no learned order; no hub, degree or IDF attribute; no text window; no question encoder; no GPU; no change of the link set; no PL8; no use of the golds in the walk; no `s1eval`; no edit of any pinned module; no use of the SEM2 R8/R9 routers (a question-conditioned **descent** router is R9's, in another session).
- **Searches after reading numbers:** none yet. Any later amendment will say what was read before it.

## 12. Amendments (dated; none changes a rule, a gate or an expectation)

None yet.

## 13. Results

None yet.
