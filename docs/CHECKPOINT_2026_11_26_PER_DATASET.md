# Checkpoint, 26 November 2026 (17:30 IST, revised 18:05 IST): what has been done, and what went wrong on each dataset

This is the user's "concrete checkpoint": a summary written from the filed records: one section per mechanism (section 2, links, walks, semantic generalization, branching, written to be critiqued) and one per dataset (section 5). **It is a draft taken while three things are still running** (GP3, R9, the UL2H 2Wiki row; section 6). Refresh it when they finish. Every number is copied from the page named beside it; nothing here is a new measurement, and nothing here was decided by this page.

The target the user set (25-26 Nov): **at least 0.95 of the questions with every gold passage or fact in the pool (ALL), at the smallest node count, as close to 2,500 distinct nodes per question as possible, with rules that are universal (one rule, all seven datasets).**

## 1. The short version

- **Four of seven datasets meet the target inside 2,500 nodes** (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki: 0.9554 to 0.9690 at 15 to 917 nodes). **Three do not** (MuSiQue 0.8846, MetaQA 0.8968, WebQSP 0.8789 at about 2,100 to 2,440 nodes). The same rule that stops at 2,500 reaches 0.95 on MetaQA at 1.99 times 2,500 nodes and on MuSiQue at 6.16 times; **on WebQSP no size read reaches it** (0.9474 at 39,530 nodes, 15.81 times).
- **No universal link-removal rule exists** in any form we declared: label and dot cells (UL3), a shared keep-fraction program (UL4), gold-path attributes (GP1), rank windows, text windows, scores, labels and per-question caps (GP2, quarter and rest of dev). The link set stays the base layout (UL2, a70ec8cc: nothing carried). The pool re-run on a final link set (PL8) has **not** been declared, because the user has not said the link set is final.
- **No depth signal exists** (UL5): no label-free statistic of five tells a question how deep its walk must go; the best saving over a uniform depth is 1.0 percent, and all 20 shuffles beat it.
- **The semantic-space work (SEM2) learns and transfers in places, but has not improved a pool or a retrieval number.** Operators learn links (R1), a branching chain beats cosine on stored links (R5), a generic operator crosses passage to KB one way (R6), but the routed scorer is below a flat question cosine on four of five datasets (R8), and the region form (R4) is not carried. On the rows that decide, the cap on branches, not the node budget, is the loss.
- **The unit was wrong for part of the search.** GP2 counted links per question; the user's bound is distinct nodes. WebQSP has 1.72 to 2.24 million candidate links per question against 39,572 nodes in its uncut pool. GP3 (declared 7c85205b, amended 52301336) counts nodes, and is running.
- **No model was retrained in this phase** (the user's ruling: models after the links are verified and unified). The GPU has been the other account's since 21 Nov (a VM holds about 17 to 20 GB of the card), so everything here ran on CPU.

## 2. How it works, by mechanism (written to be critiqued)

Each block gives the idea in plain words, the assumption under it, what was run and found, where it could be wrong ("attack points": the author's own, written against this work, not findings), and what would change the call. Every number is copied from the filed page named in section 9.

### 2.0 The pipeline and the words

```
question
  -> first stage   e3v dense + SPLADE, fused (RRF). The top results are the SEEDS.
  -> walk          from the seeds along LINKS, up to 3 hops. A CAP limits how many links one node passes on,
                   a BUDGET limits how many distinct nodes one hop may add. Candidates are ordered by cosine to the question.
  -> pool          every distinct node the walk kept. This is the count in "2,500 nodes".
  -> model         an MLP or a GNN reads the pool (not retrained in this phase).
```

| word | meaning here |
|---|---|
| node | one passage, article or KB entity; a pool is counted in distinct nodes |
| gold, ALL | the passages or facts a question needs; ALL = the pool holds every one of them. "0.95" = 95 questions in 100 |
| S, R, K | S = structure the data already has (KB triples, Wikipedia hyperlinks); R = rule links read from text (mention, name co-mention, title co-mention); K = e3v nearest neighbours. Duplicates keep S over R over K; every edge carries the label of the rule that made it |
| cap, budget | cap = most links followed out of one node; budget = most distinct nodes one hop may add. One declared setting for every question |
| oracle | a read that already knows the golds. A ceiling, never a result |
| ROOM | R8's call: a router that knew the gold would lift coverage by a lot (more than 1.0 point) |
| ABOVE, LEVEL, BELOW | the 95% interval of a paired difference lies above, across, or below its line |
| R-precision, MRR | how high a stored link's true end ranks among candidates (SEM2 reads: links, not questions) |

At a glance:

| mechanism | what it tries | status | strongest evidence | weakest point |
|---|---|---|---|---|
| 2.1 Links (S, R, K) | give the walk edges | in place; layout stays base (UL2) | 4 of 7 datasets inside 2,500 on them | one-hop precision is the wrong test; R.mention is the bulk of candidates on WebQSP and MuSiQue |
| 2.2 Link tightening | drop links a question does not need, one rule for all seven | closed, nothing carried (UL3, UL4, GP1, GP2) | needed links rank early by similarity | no attribute separates them; S protection and universality have a price |
| 2.3 Walks | caps, budgets, depth | 3 of 7 short at 2,500 | misses are all cap or budget cuts | no signal sets the depth (UL5) |
| 2.4 First stage | the seeds | per dataset | 2Wiki +10.85 | KB datasets start from weak seeds |
| 2.5 Semantic generalization | links as learned operations on frozen vectors; unseen relations; passage to KB | partial, no pool number moved | R1 learns; R3(c) +11.98 on WebQSP | read on stored links, near a floor |
| 2.6 Branching | regions, chains, routing, a descent tree | R4 no, R5 yes (stored links), R8 routed below flat cosine, R9 running | a router that knew the gold needs 1 to 2 added nodes | nothing built completes more than 60.7 percent |

### 2.1 Links (S, R, K)

- **Idea.** A walk needs edges. The three families are cheap and need no LLM and no spaCy; the label says which rule made each edge.
- **Assumption.** The links a question needs are among these three families; the label says something about whether a link is useful.
- **Ran and found.** UL2 compared layouts on e3v and carried none (the base layout stays). The ledgers (UL2P, UL2S, UL2H) measure precision and how many golds each hop reaches. At the 2,500 pick the misses are cap or budget cuts, not missing links (section 2.3). Scale differs a lot: S is 72.3 percent of MetaQA's candidate links, 52.1 percent of HotpotQA SRK's and 38.5 percent of WebQSP's (UL3); R.mention holds 75.8 percent of MuSiQue's any-link candidates and 99.9 percent of WebQSP's.
- **Attack points.** (1) One-hop precision misleads: MetaQA's S has 0.005 percent one-hop precision and is the only unit that reaches ALL 1.0 at three hops. (2) The rule links are the bulk of the candidates on the two datasets that fail worst, and they are low precision. (3) The labels are fixed by the rules, so a useful link of a useless label cannot be told apart by label alone.
- **Would change the call.** A link family that raises coverage at the same node count on the three short datasets; UL2H's remaining rows (which label would link the golds the first stage and walk miss).

### 2.2 Link tightening (UL3, UL4, GP1, GP2)

- **Idea.** The walk sees too many candidates (WebQSP: 1.72 to 2.24 million candidate links per question). Remove the ones a question does not need, with one rule for all seven datasets, from a label, a dot score, a rank or text window, or a per-question cap.
- **Assumption.** Links on gold paths differ from other links in something measurable without the answer, and the difference holds on every dataset. Two constraints are fixed by the user: universal (carried everywhere or nowhere) and S never removed (A.1). Hub, degree and IDF attributes are excluded by the user's 24 Nov ruling (hub line closed).
- **Ran and found.** UL3 (label and dot cells): the one-hop screen passes 1, 2 and 3 of 9 cells for its three forms. UL4 (one program over all cells): no rule; a one-hop filter removes at most about one candidate in seven; universality costs 12.0 points of price on K2 (the expectation was under 10). GP1 (gold-path attributes): no attribute separates the links gold paths need; but 35 to 90 percent of them sit in the top 30 of their source's links by the dot of the two endpoints, against 0.1 to 36.9 percent of controls, a higher share in every one of the nine cells (a source-relative rank, GP1 15.8; the 35 is WebQSP's, the 90 and the 36.9 are MultiHop-RAG's). GP2 (windows, scores, labels, caps; fixed on a quarter, read once on the rest of dev): the final rule keeps 0.954 to 0.999 of reach in all nine cells and fails the price line in all nine (it keeps 86 to 90 percent of links).
- **Attack points.** (1) The price line (keep at most half the links) and the reach line are mine, set before the numbers; a different line changes the call. (2) Price is counted in links, and links proved the wrong unit for WebQSP (GP3 counts nodes). (3) The families read are one-hop, windows, scores and caps; a learned path-level scorer is named, not run. (4) The A.1 protection of S costs +65 points of price on MetaQA alone.
- **Would change the call.** GP3 showing a score order that meets the bound in distinct nodes where link counts failed; the user dropping A.1 or the universality ruling (each is a choice, not a finding).

### 2.3 Walks (caps, budgets, depth)

- **Idea.** From the seeds, follow links for up to three hops, keep a capped, ordered set, stop at a node budget. One declared rule, the same for every question.
- **Assumption.** The golds lie within three links of the seeds; the question's cosine orders candidates well enough that one setting holds them; one setting suits all questions.
- **Ran and found.** PL7 (eval, the 2,500 pick): MultiHop-RAG, SQuAD, HotpotQA and 2Wiki meet 0.95 (0.9554 to 0.9690 at 15 to 917 nodes); MuSiQue 0.8846, MetaQA 0.8968 and WebQSP 0.8789 do not. The misses at 2,500 are cuts by a cap or a hop budget: MuSiQue 36 of 300 questions (39 golds, every one a cut; none ranked out, none unreachable), MetaQA 33 (375 golds, 373 cut), WebQSP 31 (81 golds, all cut). Reaching 0.95 costs 1.99 times 2,500 on MetaQA, 6.16 times on MuSiQue and is not reached on WebQSP (0.9474 at 15.81 times). An oracle that picks the depth per question needs 1,831 nodes on MuSiQue, 2,745 on MetaQA, 9,301 on WebQSP; UL5 found no label-free statistic of five that sets the depth (best saving 1.0 percent; all 20 shuffles beat it).
- **Attack points.** (1) The cap and the budgets are one global setting chosen on dev; the oracle says a per-question allocation matters and nothing found it. (2) The order is question cosine, which R8 found hard to beat, and which a cap applies to before the hop-3 and hop-4 golds appear. (3) The bound counts nodes, not cost: WebQSP's walk takes 65 to 145 seconds per question per scoring (GP3's time rows). (4) The 2,500 pick is a point on a curve, not an optimum of a fit.
- **Would change the call.** GP3 (per-hop node budgets N1 >= N2 >= N3 solved exactly, with a PageRank order) meeting 2,500 on MuSiQue, MetaQA or WebQSP; R9's cap ladder.

### 2.4 First stage (the seeds)

- **Idea.** The walk starts from the first stage's top results (e3v dense and SPLADE, fused).
- **Assumption.** A gold, or a close neighbour of one, is among the seeds.
- **Ran and found.** e3v minus frozen at R@5: SQuAD -4.00, MuSiQue +1.97, HotpotQA -0.43, 2Wiki +10.85, WebQSP -4.22 (and -14.12 at R@1000), MetaQA -0.22, MultiHop-RAG -2.62 (the windows win +4.94 back, E6). On the KB datasets the first stage's coverage of ALL golds at depth 1,000 is 18.70 percent (MetaQA) and 25.15 percent (WebQSP); 83.5 percent of WebQSP's missed golds are one hop from a seed.
- **Attack points.** (1) Published WebQSP systems are given topic entities; ours are retrieved from text, so the pool starts from poor seeds. (2) The R@5 gap to HippoRAG 2 on MuSiQue (66.3 against 74.7) is the first stage (56.4 against 69.7). (3) The encoder is a settled ruling (e3v, D101); the first-stage gap is the vector work that comes after the links.
- **Would change the call.** A stronger first stage on the three short datasets; a seed rule that uses entities (a scope question for the user, not a finding).

### 2.5 Semantic generalization (SEM2 R1, R2a, R3, R6, R7)

- **Idea (the user's, 18 Nov).** Do not fine-tune an encoder. Give every link label one operator, a rotation in an enlarged space, learned from the links alone on any frozen vectors, so a link is "an operation", the same machinery covers any graph, a relation never seen gets its operator from its name's vector, and a passage graph and a KB can share operators.
- **Assumption.** A rotation per label captures what the link means; a name carries enough meaning for an unseen relation; one generic operator crosses from passages to a KB.
- **Ran and found.**
  - R1 (do operators learn links?): **LEARNS**; a rotation beats plain cosine at ranking a stored link's true end (WebQSP +39.07 pooled, MetaQA +12.31).
  - R2a (the traversal inside the first stage, nothing learned): **PARTIAL**; what the operators add at R@5 is +0.11 on SQuAD and +0.06 on MetaQA (ABOVE), LEVEL on four, and -0.22 on MuSiQue (BELOW); most of the traversal's gain is the seeds' own cosine neighbourhood.
  - R3(c) (an unseen relation, from its name): **PARTIAL**; WebQSP +11.98 [+8.99, +15.23] over cosine, but a shuffled name already gives +7.04 and sits 4.94 below the right name, so more than half the gain is "any operator beats cosine"; MetaQA is NO on its 7 relations (a statement about power).
  - R6 (passage to KB): **ONE_WAY**; passage to KB +1.53 [+0.68, +2.52] with a generic operator; KB to passage -1.37 BELOW; the floor MRR is 0.4 to 2.5.
  - R7 (links withheld, fill them): **FILL_CARRIES, small**; a fill beats no fill on 6 of 6 (+0.62 to +4.31 R-precision points), on pairs the fit could not have drawn it is ABOVE on 4 (+0.19 to +0.58); it recovers 2 to 17 percent of what withholding half the links costs.
- **Attack points.** (1) R1 to R7 rank a stored link's true end among candidates. They test whether operators model the links, not whether a question's gold is found; only R8 and R9 add the question. (2) Several reads sit near a floor, where a point is a few nodes. (3) **No pool or retrieval number has moved.** The question-level read of R6 (does the operator-moved seed find a gold the walk misses) is named, not run. (4) Part of every gain is "any operator beats cosine".
- **Would change the call.** A question-level read where an operator-moved seed finds a gold the stored walk misses; R9 using the operators in a descent.

### 2.6 Branching (SEM2 R4, R5, R8, R9)

- **Idea (the user's, 25 and 26 Nov).** One point cannot stand for many nodes: the neighbours of a node spread out like a tree. Operate on a vector, get a region, and descend branch by branch with the question instead of walking and backtracking (a question-conditioned semantic b-tree).
- **Assumption.** The question can pick the right branch at each node with a cheap score, so a thin path can be walked.
- **Ran and found.**
  - R4 (a region instead of a point): **NOT_CARRIED**; region minus point is BELOW on 5 of 7, LEVEL on 2, ABOVE on none (the point beats cosine by +5.2 to +30.7 at 15 to 49 R-precision).
  - R5 (a second hop through branching operators): **BRANCH_CARRIES 6/6**, +1.12 to +7.68 R-precision points (four of them at 2.5 to 4.2), on stored links.
  - R8 Part A: **ROOM on 5 of 7**. A router that knew the gold needs a median of 1 to 2 added nodes, and would lift the first stage's ALL coverage from 65.38 to 98.17 (MuSiQue), 18.70 to 85.10 (MetaQA), 25.15 to 98.67 (WebQSP), 83.82 to 99.87 (HotpotQA), 47.66 to 99.93 (2Wiki). SQuAD and MultiHop-RAG: NO_ROOM.
  - R8 Part B: **NOT_CARRIED**; the routed scorer is BELOW a flat question cosine on 4 of 5 (-8.75 MuSiQue, -7.33 MetaQA, -5.83 WebQSP, -9.92 HotpotQA; +14.50 on 2Wiki, which comes from the stored label, since a control with another question is level). The best completion of any scorer is 60.7 percent (HotpotQA), 44.7 (MetaQA), 36.7 (WebQSP), 35.7 (2Wiki), 31.6 (MuSiQue), against 100 for a router that knew the gold; the cap of 100 branches, not the node budget, is the loss.
  - R9 (running): lift the cap in steps, then descend a question-conditioned tree.
- **Attack points.** (1) ROOM says room exists, not that a router reaches it (R8's own words: "H is not a gain"). (2) The hard part is the field: one gold among 229 (MetaQA) to 14,862 (WebQSP) hop-1 nodes; a flat question cosine already reads the question (beats random by 7.7 to 44.6 points), so a branch router has to beat that baseline, and so far does not. (3) R5's gains are on stored links, and four of its six are between 2.5 and 4.2 points. (4) R4 tried one region shape (a trained box is named, not run).
- **Would change the call.** R9 showing that lifting the cap, or a descent, completes clearly more than 60.7 percent or beats the flat question cosine on at least 3 of 5.

### 2.7 What ties them (a reading, not a filed result)

1. Where we can see why a gold is missed at the 2,500 pick, it is a cap or budget cut (every gold on MuSiQue and WebQSP, 373 of 375 on MetaQA); on MuSiQue none is ranked out and none unreachable (PL7).
2. The links gold paths need are not unusual: no attribute separates them (GP1, GP2). They do rank early by similarity to their source (above the controls in all nine cells). So the information to find them is mostly closeness, and the lever is **order under a cap**, not a better filter.
3. A thin path exists (oracle depth 1,831 on MuSiQue; a router that knew the gold needs 1 to 2 added nodes) and nothing built finds it (UL5: no depth signal; R8: best completion 60.7 percent, routed scorer below flat cosine on 4 of 5).
4. Three kinds of dataset: text corpora with small pools (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki: inside the bound); MuSiQue (hop-3 and hop-4 golds behind low-precision mention links); typed KBs (MetaQA, WebQSP), where S is both the lever and the cost and the hop-1 field is 229 to 14,862 nodes.
5. The semantic work fits this picture: it improves how links rank on stored links and has not yet shown a question-level gain.

### 2.8 Questions to critique, most plan-changing first

1. **Is ALL at 0.95 the right yardstick?** It needs every gold; a question with three golds fails if one is missing. No model was retrained in this phase, so what a pool at 0.88 against 0.96 does to answer quality is unmeasured here.
2. **Is distinct node count the right cost?** WebQSP's walk takes 65 to 145 seconds per question per scoring; a pool that meets the node bound may cost more than it saves. Link counts (GP2) and node counts (GP3) gave different pictures.
3. **Is "universal" too strict for seven datasets that split into text corpora and typed KBs?** It is the user's 21 Nov ruling; it is listed because the evidence splits (rules that cut HotpotQA and 2Wiki do not cut the others; UL4 prices universality at 12.0 points), not as a proposal to drop it.
4. **Is "S never removed" right?** It is an assumption (A.1) and was priced at +65 points of price on MetaQA; UL3's premise for it (S is small) was withdrawn.
5. **Hub, degree and IDF attributes are excluded** by the user's 24 Nov ruling; every conclusion in 2.2 holds for the attributes read, not for those.
6. **R.mention is the bulk of the candidates** on WebQSP (99.9 percent) and MuSiQue (75.8 percent). Is the problem how many rule links exist, or how they are ordered? GP1 points at order.
7. **Seeds on the KB datasets.** Published WebQSP systems start from given topic entities; we start from retrieved text and cover 18.70 and 25.15 percent of golds at depth 1,000. Is that scope what we want to claim?
8. **Optimism.** PL7's pick was fixed on dev and read once on eval; GP1 and GP2's rules were fixed on a quarter of dev and read once on the rest of dev; SEM2 reads use their own declared splits. The pools that sit close to the line (SQuAD 0.9554, Wilson lower end 0.9515) have little margin.
9. **Floor effects in SEM2.** R6's +1.53 sits on an MRR floor of 0.4 to 2.5; R7's control gains are +0.19 to +0.58. A point there is a few nodes.
10. **The oracles are ceilings.** Oracle depth and the perfect router use the golds. They bound what to hope for; they do not say it can be reached label-free.

## 3. What has been done, in order (filed records)

| stage | what it did | filed |
|---|---|---|
| Encoder (E0 to E7) | screened encoders; D101 on 21 Nov: **e3v** (gte-modernbert + granite-small, 1,152 wide, 196.7M parameters) is the encoder; frozen kept as the comparison; vectors stored as **int8** (E5: 1,152 bytes per document, 0.38 of frozen's, free on six datasets; E7: the windows at 50 percent of exact) | E3W 4e8568a, E5/E4 87e24f9, E7 2379f96 |
| MultiHop-RAG joins (7th dataset) | own package, e3v encode, link grade, windows | MHR1 ac2b538, UL2 section 13 d20181a, E6 28f9170 |
| Links on e3v (UL1, UL2, ledgers) | S (structure: KB triples, hyperlinks), R (rule links: mention, name and title co-mention), K (kNN), deduplicated S > R > K, every edge labelled by its rule; UL2P/UL2S/UL2H precision and hop-coverage ledgers | UL2 a70ec8cc NONE_CARRIED (layout = base); ledgers c8cc686 (MHR, SQuAD) |
| Pools (COV, PL1 to PL7) | pools on the UL2 layout; the coverage target; rounds that lifted and then restricted the node bound back toward 2,500 under one rule | PL2 efbcf3c, PL3 2824c2e, PL4 18cccdc1, PL7 22d12ea2 **NOT_CARRIED** (7 of 7 level) |
| Link tightening | UL3 (one-hop screen) FAILS_SCREEN; UL4 (universal program) no rule; UL5 (depth signal) NO_SIGNAL; GP1 (gold-link patterns) NOT_UNIVERSAL; GP2 (windows, scores, caps) NOT_UNIVERSAL | UL3 a3a87915, UL4 619d2e23, UL5 32b3d4a5, GP1 838f199a + 7a5c0577, GP2 103d36cf + 7a5c0577 |
| Semantic space | SEM1 NOT_HELD; SEM2 R1 to R8 (operators, traversal, unseen relations, regions, branching, passage-KB bridge, withheld links, routing); hub line **closed** by the user (8.15) | SEM1 4280819; R1 LEARNS; R2a 9397f07; R3(c) 9a22b70f; R4 a6d02a27; R5 72ad112f; R6 3adea6cc; R7 f7f9f95d; R8 c6816eca + 8eee96d0 |
| Running now | GP3 (distinct nodes, with a PageRank order), R9 (cap ladder, descent router), UL2H's last host row | section 6 |

## 4. The seven datasets side by side

Pool at the 2,500 pick, eval, frozen lists (PL7 section 14.2) and the cost of reaching 0.95:

| dataset | links it has | ALL at the 2,500 pick [Wilson 95%] | mean nodes | meets 0.95 inside 2,500 | nodes to reach 0.95 | uncut pool / oracle depth (UL5) |
|---|---|---|---|---|---|---|
| MultiHop-RAG | R, K (no S) | 0.9690 [0.9572, 0.9776] | 50.0 | yes | 50 | 255 / 35 |
| SQuAD | R, K (no S) | 0.9554 [0.9515, 0.9589] | 15.0 | yes | 15 | 15 / 15 |
| HotpotQA | S (hyperlinks), R, K | 0.9634 [0.9589, 0.9674] | 77.9 | yes | 78 | 85 / 41 |
| 2Wiki | S (hyperlinks), R, K | 0.9609 [0.9573, 0.9641] | 917.0 | yes | 917 | 911 / 541 |
| MuSiQue | R, K (no S) | 0.8846 [0.8712, 0.8967] | 2,438.2 | **no** | 15,400 (0.9640; 6.16 x) | 20,038 / 1,831 |
| MetaQA | S (KB triples), R, K | 0.8968 [0.8906, 0.9027] | 2,335.9 | **no** | 4,973 (0.9582; 1.99 x) | 4,984 / 2,745 |
| WebQSP | S (KB triples), R, K | 0.8789 [0.8614, 0.8944] | 2,094.6 | **no** | none read (0.9474 at 39,530) | 39,572 / 9,301 |

The final GP2 rule (per-question share cap, 0.9 / 0.9 / 0.5) on the rest of dev, reach / price (price = share of candidate links kept; the line is 0.50), GP2 section 18:

| dataset | reach | price | | dataset | reach | price |
|---|---|---|---|---|---|---|
| MultiHop-RAG | 0.9993 | 0.9000 | | HotpotQA RK / SRK | 0.9825 / 0.9952 | 0.8829 / 0.8994 |
| SQuAD | 0.9645 | 0.8547 | | 2Wiki RK / SRK | 0.9797 / 0.9865 | 0.8734 / 0.8725 |
| MuSiQue | 0.9659 | 0.8718 | | MetaQA | 0.9537 | 0.8760 |
| WebQSP | 0.9732 | 0.9004 | | | | |

It keeps 86 to 90 percent of every cell's links, so it fails the price line in all nine cells.

## 5. Per dataset: what we did and what went wrong

### 5.1 MultiHop-RAG (the seventh dataset, added 21 Nov)

- **What it is.** An article-node corpus of 609 nodes; no S, so only R and K links and no SRK cell. 224 dev questions. Its retrieval cache is laptop-only on this setup, so every row that needs it ran on the laptop, not the host.
- **Where it stands.** The pool holds ALL 0.9690 at 50 nodes: inside the bound by a wide margin.
- **What we found.**
  - First stage: e3v is BELOW frozen on d5 (-2.62, MHR1 ac2b538); the windows (E6) win it back and more (d5 64.29 to 69.23, +4.94; +2.32 over frozen, 28f9170).
  - Links: each UL2 form alone over the base layout is ABOVE on eval (H5 +6.26, HK2 +7.03, H10 +7.83); the e3v rule links are 1.88 BELOW frozen's, so the links were not carried before H. UL3: the C and PC variants pass the screen here (price 45.4 and 47.8 at reach 97.1 and 97.3).
  - GP2: no rule of the set puts it in (the final rule's price is 0.9000 at reach 0.9993).
  - SEM2: R8 says NO_ROOM (the corpus has fewer nodes than the first stage's depth of 1,000, so coverage is already 100.00); R5 has zero units; R1h's hub fix H2 was 7.28 BELOW.
- **Issues.** (1) Everything reads at the ceiling because the corpus is smaller than the first-stage depth, so MultiHop-RAG cannot discriminate between rules. (2) A tiny graph: 94 questions on GP2's rest-of-dev read. (3) Laptop-only inputs.
- **Open.** Nothing.

### 5.2 SQuAD

- **What it is.** One passage corpus, R and K links only. 1,498 dev questions.
- **Where it stands.** ALL 0.9554 at 15 nodes (the pick is the first-stage list alone). The lowest margin of the four that pass: the Wilson lower end is 0.9515.
- **What we found.**
  - First stage: e3v is 4.00 points BELOW frozen on R@5 (order, not reach). The windows lift d5 from 83.69 to 89.01 (E6).
  - Links: UL2H shows no gold is isolated: every unreached gold is far, so the lever is ranking and the cap, not a missing link (K.knn at 2 hops reaches 95.19 percent at 77.9 nodes). GP2's final rule keeps 85.5 percent of links.
  - SEM2: R2a adds a little (T2 +0.11); R5's chain beats cosine most here (+7.68 R-precision points); R8: NO_ROOM (cov0 99.60, H 0.33).
- **Issues.** The loss is in the first stage (the encoder), not the links; pools sit close to the line.
- **Open.** Nothing in this phase; the first-stage gap is the vector work that comes after the links.

### 5.3 HotpotQA (S = hyperlinks; shown with and without S)

- **What it is.** The largest source set (about 5.2 million source rows). Two versions, with and without S (the user's ruling 21 Nov). 1,508 dev questions.
- **Where it stands.** ALL 0.9634 at 77.9 nodes.
- **What we found.**
  - First stage: e3v is level (-0.43).
  - Links: UL3 does not shrink the RK version (best price 62.2 percent); the SRK version cannot pass under "S never removed" because S holds 52.1 percent of the candidates and 99.9 percent of the covered slots. GP1: 78 to 83 percent of the links gold paths need sit in the top 30 of their source's links, against 0.1 to 0.2 percent of controls; 60 to 68 percent have an exact title match. GP2's cutting rules do cut it (to 6,346 to 8,648 links per question) because a cell's own optimum is only 119 to 193 links per question (HotpotQA RK and 2Wiki RK together), but they are not universal.
  - SEM2: R8 ROOM (cov0 83.82 to 99.87 with at most 1,500 added nodes; 95.1 percent of misses are one hop out); the flat question cosine completes 60.7 percent, the best of any dataset; R4's region is 1.80 BELOW the point; R5 +1.34.
- **Issues.** (1) A cost problem: the kNN groups had to be rebuilt from the record's pointer index (text groups differ in 16 of 5.2 million rows), and the gate row for e3v's K took 85,760 s on the degraded host (about 2 hours when healthy). (2) The cap of 100 branches, not the budget, is the loss on the routing read (92 to 99 percent of candidate recall).
- **Open.** UL2H's HotpotQA read is waiting on the last host row.

### 5.4 2Wiki (S = hyperlinks; shown with and without S)

- **What it is.** Two versions, with and without S. 1,496 dev questions.
- **Where it stands.** ALL 0.9609 at 917 nodes (PL2 needed 2,140; PL3 cut it to 917).
- **What we found.**
  - First stage: e3v is +10.85 ABOVE frozen on R@5, the biggest gain of the seven.
  - Links: the one dataset where UL3's screen passes on every cell (cuts 52 to 64 percent of candidates for 2 to 6 points of reach); GP1: 82 to 83 percent of needed links in the top 30 of their source's links against 0.1 percent of controls; 78 to 82 percent exact title. GP2's A2 and P5 rules put it in (the same 119 to 193 links per question own optimum), and no rule of that set puts MultiHop-RAG, MuSiQue, SQuAD or MetaQA in.
  - SEM2: R8 ROOM (cov0 47.66 to 99.93; 98.3 percent of misses one hop out); of all datasets the routed scorer helps only here (+14.50), and a control with another question is level, so the gain is the stored label's, not the question's. R4's region is 2.27 BELOW.
- **Issues.** The slowest row of the phase: R9's 2Wiki cap row takes about 8 minutes per question (75 questions, ends near 03:00 IST). GP1's extraction rows took 1.4 hours.
- **Open.** R9 cap read and UL2H's 2Wiki read (the last host row).

### 5.5 MuSiQue (no S; the hard passage set)

- **What it is.** 117,534 nodes; R and K links only; 2, 3 and 4 hop questions. 1,534 dev, 2,417 eval questions.
- **Where it stands.** **Short of the target at 2,500: ALL 0.8846.** The curve has no cliff: 0.8999 at 3,449 nodes, 0.9222 at 5,742, 0.9350 at 9,076, 0.9429 at 13,615, then **0.9640 at 15,400 (6.16 times 2,500)**. The uncut pool needs 20,038 nodes; **an oracle that picks the depth per question needs 1,831**.
- **What we found.**
  - Misses at 2,500: 36 of 300 questions, 39 golds, every one cut by a cap (hop-2 cap 18, hop-2 budget 9, upstream 12), none ranked out and none unreachable.
  - First stage: e3v is +1.97 ABOVE frozen at R@5, but the pools lose 0.42 golds and the models give it back.
  - Links: UL3 cannot shrink it (price 86 to 99.5 percent); R.mention holds 75.8 percent of the any-link candidates and is below the any-link set in one-hop precision. GP1: 56 percent of the links gold paths need are in the top 30 of their source's links (1.8 percent of controls), and 43 percent have an exact title match (1.8 percent). GP2: no universal setting cuts MuSiQue.
  - UL5: the oracle depth (1,831) is below 2,500, so a depth signal would pay here, but none of five statistics found one (closure of the gap to the oracle: 0.107).
  - SEM2: R8 ROOM (cov0 65.38 to 98.17; cost* median 1 added node); the routed scorer is 8.75 points BELOW the flat question cosine; R4's region 3.66 BELOW; R5 +3.14; R2a T2 BELOW.
  - HippoRAG comparison: our pools hold +0.9 [+0.6, +1.2] more golds than HippoRAG 2's walk at equal rows (COV A1, 25ce7d7; that walk used our link families); the R@5 gap to HippoRAG 2 (66.3 against 74.7) is the first stage (56.4 against 69.7).
- **Issues.** (1) Three- and four-hop golds are cut by the per-frontier cap before any budget binds. (2) The depth is a per-question allocation problem with no label-free signal. (3) Mention links are numerous and low-precision.
- **Open.** GP3's PageRank order, R9's cap ladder.

### 5.6 MetaQA (S = KB triples)

- **What it is.** A KB graph of 43,234 nodes; 1 to 3 hop typed questions. 1,497 dev, 9,785 eval.
- **Where it stands.** **Short at 2,500: ALL 0.8968.** 0.9240 at 3,475 nodes; **0.9582 at 4,973 (1.99 times 2,500)**. Uncut pool 4,984 nodes; oracle depth 2,745.
- **What we found.**
  - The first stage finds almost nothing (frozen dense R@5 0.87; GP1: 14 of 3,228 golds); 87 percent of the missed golds are two or three hops out.
  - Misses at 2,500: 33 of 300 questions, 375 golds (11.4 per question), 373 cut by a cap or the hop-3 budget (264 by the hop-3 budget).
  - Links: S is 36 percent of the pairs, but **72.3 percent of the candidates**, and has the lowest one-hop precision (0.005 percent) while being the only unit that reaches ALL 1.0 at three hops. UL2's first KB rule text made F0T 56 times the filed one (706,154 against 12,605 pairs), which broke the size bound until 12.6 rebuilt it from names. UL3 cannot pass under "S never removed"; the cost of that protection is +65 points of price here (UL4: own optimum 17 with S free, 83 with S protected). GP1: MetaQA is the odd one out (its dot, direction and exact-title AUCs lie below 0.5; half of the needed weight is on S).
  - SEM2: R8 ROOM (cov0 18.70 to 85.10 with 1,500 added nodes; the budget costs 29 percent of recall here); R3(c): NO on its 7 relations (all MRR below 2.2, a statement about power, not a finding that it cannot transfer); R4 region 1.02 BELOW; R5 +1.94.
- **Issues.** (1) The golds are typed multi-hop paths the first stage never sees. (2) S is both the lever and the cost, so no rule that protects S can be cheap. (3) The pool needs the three-hop budget.
- **Open.** UL2H's MetaQA read, GP3 (S-only walk reference).

### 5.7 WebQSP (S = KB triples; the explosion)

- **What it is.** A KB graph of 2,592,894 nodes; typed one- and two-hop questions. 1,507 dev, 1,503 eval.
- **Where it stands.** **Short at every size we read.** ALL 0.8789 at 2,095 nodes; 0.8975 at 4,977; 0.9175 at 6,080; 0.9328 at 10,817; 0.9421 at 20,738; the PL6 pick **0.9474 at 39,530 (15.81 times 2,500) does not reach 0.95**. The uncut pool is 39,572 nodes (ALL 0.956); **the oracle depth is 9,301, 3.7 times 2,500**: no depth rule, with or without labels, reaches 2,500.
- **What we found.**
  - Misses at 2,500: 31 of 300 questions, 81 golds, all cut by caps or the hop-2 and hop-3 budgets.
  - First stage: e3v is 4.22 BELOW on R@5 and 14.12 BELOW on R@1000 (dense R@1000 only 36.7 percent): the lever is the links, not the first stage.
  - Links: S is 1.2 percent of the pairs, **38.5 percent of the candidates**, and carries 55 percent of the weight of the links gold paths need (the controls: 1.7 percent). R.mention holds 99.9 percent of the any-link candidates. 83.5 percent of the golds the first stage misses are one hop out, in a field of 14,862 neighbours. UL3's price stays at 55.2. GP2: WebQSP is the only dataset a universal absolute cap can put in (C2 keeps 95,611 links per question on the quarter and 114,505 on the rest, 38 to 46 times 2,500), from 1.72 to 2.24 million candidate links per question.
  - SEM2: R1's operators gain +39 pooled (one-to-one relations, not hubs); R3(c) transfers to unseen relations on WebQSP (+11.98 [+8.99, +15.23]); R8 ROOM (25.15 to 98.67); best completion of any scorer 36.7 percent; R4 region 0.61 BELOW; R5 +1.12.
  - Literature (WebQSP): published systems start from topic entities given in the data and take a typed subgraph within about two hops (about 4,000 triples), then learn a question-versus-relation or triple scorer; their metrics are answer or triple recall at a fixed K (SubgraphRAG: 0.883 shortest-path triple recall at K = 100), not "every gold in the pool". We do not have given topic entities, and our R and K links make the graph far larger than theirs.
- **Issues.** (1) The candidate set is 40 times the node count of the pool; the explosion is our own R and K links on a graph that is already typed. (2) The walk itself is expensive: GP3's time rows take 65 to 145 seconds per question per scoring on this dataset. (3) The first stage is weak, so the pool starts from poor seeds.
- **Open.** GP3 (S-only walk, PageRank order, question-versus-triple text scores named), R9's WebQSP cap row (question 40 of 75 at 17:02).

## 6. What is still running or undecided

| item | state | what it decides |
|---|---|---|
| GP3 (docs/GP3_NODE_COUNT_RULE.md) | declared 7c85205b; amendment A1 52301336 adds a PageRank order; six time rows on the host (SQuAD, MuSiQue, MetaQA, WebQSP done; HotpotQA running, about 150 s per question at 17:30; 2Wiki queued behind it); a separate gp3c-webqsp row was cancelled at 15 minutes (the GP1 session's row, reason not read); section 13 still "None yet" | the least worst-dataset mean distinct nodes at ALL 0.95, exact, with a shuffle null; whether a PageRank or GP2 score order meets 2,500 anywhere it currently does not |
| R9 (SEM2 section 17) | cap rows done on MuSiQue, MetaQA, HotpotQA; WebQSP running (question 50 of 75 at 17:23) and 2Wiki running (question 10 of 75 at 17:04, about 556 s per question, so it ends near 03:00 IST); tree rows done on MuSiQue and MetaQA, the others queued; nothing of R9 is filed | whether lifting the 100-branch cap, and then a descent through a question-conditioned tree, finds the thin path the oracle uses |
| UL2H (hop-coverage ledger) | 2Wiki row running (s1eval question 4,259 of 12,576 at 17:28); five host rows unread | which label or form would link the golds the first stage and walk miss, per dataset |
| PL8 (the one pool re-run) | **not declared** | declared only when the user says the link set is final |
| int8 world, deletes | not started; the delete list needs the user's per-action OK | after the pools are final |
| Models and the unified refit | not started | after links, pools and the int8 world |

## 7. Issues common to all seven

1. **No universal one-hop link rule** (UL3, UL4, GP1, GP2). Where a rule cuts, it cuts one family of datasets (2Wiki, HotpotQA, WebQSP by an absolute cap) and no rule cuts MultiHop-RAG, MuSiQue, SQuAD or MetaQA. A label-and-dot one-hop filter removes at most about one candidate in seven (UL4).
2. **One-hop precision is the wrong test**: a label with 0.005 percent one-hop precision (MetaQA's S) is the only one that reaches ALL 1.0 at three hops. Link tests must be path-level.
3. **Links versus nodes.** A link rule's price is not a node count; GP3 was declared to measure nodes.
4. **The cap, not the budget**, is the loss on the routing read for MuSiQue, HotpotQA and 2Wiki (completion is 92 to 99 percent of what the candidate set holds). The question is read (its cosine beats random by 7.7 to 44.6 points), but no scorer completes more than 60.7 percent of a population that a perfect router completes in full.
5. **The first stage differs by dataset** (e3v: +10.85 on 2Wiki, +1.97 MuSiQue, level HotpotQA; BELOW on SQuAD, WebQSP, MultiHop-RAG, MetaQA).
6. **Compute.** The GPU has been the other account's since 21 Nov. Host outages (Modern Standby) and a degraded host cost days on some rows (HotpotQA's knn rebuild, 2Wiki's GPU knn).
7. **Corrections made along the way (all disclosed on their pages):** UL3's A.1 premise (S small) was wrong at the candidate level and was withdrawn; UL4's first depth run read the wrong table cells; GP1's "question's own rank" wording was wrong (the column is source-relative, section 15.8); GP2's first WebQSP candidate figure was 391,000 and is 1.72 million; R8's XA2 expectation (misses lie beyond two hops) failed; PL7's declaration pointed at the wrong section for its results.

## 8. What this page does not say

It does not say the 2,500-node target is out of reach for MuSiQue or MetaQA (an oracle depth meets it on MuSiQue at 1,831), nor that WebQSP cannot be reached by a different walk (GP3's order, the S-only walk, and learned question-versus-relation scores are the named routes). It does not say the semantic space is closed (R9 is running). The numbers are development numbers on the current links; the pools are provisional until PL8.

## 9. Sources

PL7 section 14 (the pool table and the ladders); UL5 sections 11.2 and 11.6 (uncut and oracle depth); UL4 section 10; UL3 section 14; GP1 sections 15.1, 15.7, 15.8; GP2 sections 17 and 18; GP3 section 12; SEM2 sections 7 to 15 (R1 to R7: the operators, the unseen relation, regions, branching, passage and KB, withheld links), 16.12 and 16.14 (R8) and 17 (R9, declared); GP3 sections 12 and 13; ENC E6 (windows); UL2H ledger; ENC section 20.11 and E3W 19.8 (first stage); the project's progress record for the shas of PL2 to PL4, UL2 and the SEM2 rounds.
