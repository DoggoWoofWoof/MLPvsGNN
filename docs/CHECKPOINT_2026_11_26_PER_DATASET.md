# Checkpoint, 26 November 2026 (17:30 IST, revised 20:55 IST: CR1 added): what has been done, and what went wrong on each dataset

This is the user's "concrete checkpoint": a summary written from the filed records: one section per mechanism (section 2, links, walks, semantic generalization, branching, written to be critiqued) and one per dataset (section 5). **It is a draft taken while three things are still running** (GP3, R9, the UL2H 2Wiki row; section 6). Refresh it when they finish. Every number is copied from the page named beside it; nothing here is a new measurement, and nothing here was decided by this page.

The target the user set (25-26 Nov): **at least 0.95 of the questions with every gold passage or fact in the pool (ALL), at the smallest node count, as close to 2,500 distinct nodes per question as possible, with rules that are universal (one rule, all seven datasets).**

## 1. The short version

- **Four of seven datasets meet the target inside 2,500 nodes** (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki: 0.9554 to 0.9690 at 15 to 917 nodes). **Three do not** (MuSiQue 0.8846, MetaQA 0.8968, WebQSP 0.8789 at about 2,100 to 2,440 nodes). The same rule that stops at 2,500 reaches 0.95 on MetaQA at 1.99 times 2,500 nodes and on MuSiQue at 6.16 times; **on WebQSP no size read reaches it** (0.9474 at 39,530 nodes, 15.81 times).
- **The gold links are there; the price is the problem** (uncapped, UL2H and UL2S, section 4). Within three links of the seeds every dataset read holds at least 99.72 percent of its golds, and no gold is isolated on any arm. MuSiQue's golds: 50.36 percent are seeds, **77.84 percent one link out (528 nodes)**, 95.64 percent two links out (**24,076 nodes**, 9.6 times 2,500), 99.72 percent three out (32,101). About a fifth of MuSiQue's golds is not one link from a seed, and 22.41 percent of the golds the first stage misses have links but none to a seed or to another gold. 2Wiki is not read yet.
- **What the models were given when MuSiQue reached 78** (SWP Part D, 8 Nov; section 4): the oracle's direct gold links, not a rule. Adding the 1,090 chain pairs (each sub-question's passage to the earlier passages it names) to the rule graph lifted both models of record in-domain, 69.0 to 79.6 and 67.4 to 78.2 R@5 (+10.7 and +10.8); decoy and swapped-partner edges lowered it by 1.0 to 1.7 points, so the lift is the right connection. **The current rule links hold 49.76 percent of MuSiQue's chain slots directly, at 0.70 percent precision** (about 1 link in 140 at a gold is its partner). The same ledger (UL2P) gives, for gold pairs: MetaQA 13.12, WebQSP 19.07, 2Wiki 43.76 to 45.80, HotpotQA 65.31 to 80.01, MultiHop-RAG 85.55 percent recall, at 0.01 to 1.77 percent precision. How many of these links survive inside a 2,500-node pool is not measured.
- **Chains can be recreated from the links on the passage datasets, and the question orders the next hop better than the pair's own dot on four of six** (CR1, ee7ced71; whole graph, no pool). MuSiQue's oracle chain slots are 49.76 percent direct and **84.20 percent within two links** (median 3 intermediates; by hop count 90.8, 63.3 and 54.1 percent of questions with every slot within two links); HotpotQA 82.04 (RK) and 86.64 (SRK), 2Wiki 65.81. Ordering a source's links by the question's cosine puts the next hop ahead of the pair's dot by +3.92 (MultiHop-RAG), +8.32 (MuSiQue), +6.87 (HotpotQA) and +0.82 (2Wiki) points at rank 30, level on WebQSP (+3.00), and **25.94 points behind on MetaQA**, where the golds are answer names. No fixed fusion is safe, and no weight rule is declared. The residual order (question minus the source's direction) is below the question cosine on all six.
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
- **Ran and found.** UL2 compared layouts on e3v and carried none (the base layout stays). The ledgers (UL2P, UL2S, UL2H) measure precision and how many golds each hop reaches. At the 2,500 pick the misses are cap or budget cuts, not missing links (section 2.3). Uncapped, every dataset read holds at least 99.72 percent of its golds within three links of the seeds (section 4); the loss is the price of reaching them, not a missing link. Scale differs a lot: S is 72.3 percent of MetaQA's candidate links, 52.1 percent of HotpotQA SRK's and 38.5 percent of WebQSP's (UL3); R.mention holds 75.8 percent of MuSiQue's any-link candidates and 99.9 percent of WebQSP's.
- **Attack points.** (1) One-hop precision misleads: MetaQA's S has 0.005 percent one-hop precision and is the only unit that reaches ALL 1.0 at three hops. (2) The rule links are the bulk of the candidates on the two datasets that fail worst, and they are low precision. (3) The labels are fixed by the rules, so a useful link of a useless label cannot be told apart by label alone.
- **Would change the call.** A link family that raises coverage at the same node count on the three short datasets; UL2H's remaining rows (which label would link the golds the first stage and walk miss).

### 2.2 Link tightening (UL3, UL4, GP1, GP2)

- **Idea.** The walk sees too many candidates (WebQSP: 1.72 to 2.24 million candidate links per question). Remove the ones a question does not need, with one rule for all seven datasets, from a label, a dot score, a rank or text window, or a per-question cap.
- **Assumption.** Links on gold paths differ from other links in something measurable without the answer, and the difference holds on every dataset. Two constraints are fixed by the user: universal (carried everywhere or nowhere) and S never removed (A.1). Hub, degree and IDF attributes are excluded by the user's 24 Nov ruling (hub line closed).
- **Ran and found.** UL3 (label and dot cells): the one-hop screen passes 1, 2 and 3 of 9 cells for its three forms. UL4 (one program over all cells): no rule; a one-hop filter removes at most about one candidate in seven; universality costs 12.0 points of price on K2 (the expectation was under 10). GP1 (gold-path attributes): no attribute separates the links gold paths need; but 35 to 90 percent of them sit in the top 30 of their source's links by the dot of the two endpoints, against 0.1 to 36.9 percent of controls, a higher share in every one of the nine cells (a source-relative rank, GP1 15.8; the 35 is WebQSP's, the 90 and the 36.9 are MultiHop-RAG's). GP2 (windows, scores, labels, caps; fixed on a quarter, read once on the rest of dev): the final rule keeps 0.954 to 0.999 of reach in all nine cells and fails the price line in all nine (it keeps 86 to 90 percent of links).
- **Attack points.** (1) The price line (keep at most half the links) and the reach line are the author's choices, set before the numbers; a different line changes the call. (2) Price is counted in links, and links proved the wrong unit for WebQSP (GP3 counts nodes). (3) The families read are one-hop, windows, scores and caps; a learned path-level scorer is named, not run. (4) The A.1 protection of S costs +65 points of price on MetaQA alone.
- **Would change the call.** GP3 showing a score order that meets the bound in distinct nodes where link counts failed; the user dropping A.1 or the universality ruling (each is a choice, not a finding).

### 2.2a Chains and the order of links (CR1)

- **Idea (the user's, 26 Nov).** Measure how far the oracle's chain slots are from linked, recreate the chains label-free, and read how and why one order of a source's links is preferred over another by dataset ("take a weighted score or something").
- **Assumption.** A gold slot with no direct link is often joined by a short path through a shared node, and the next hop is found better by the question than by the two endpoints' own dot.
- **Ran and found** (whole graph, no pool; six datasets, SQuAD has no slot). *Part A:* a slot is D (linked), T (a node linked to both) or N. Two links reach 99.75 percent of MultiHop-RAG's slots, 84.20 of MuSiQue's chain slots, 82.04 to 86.64 of HotpotQA's, 64.35 to 65.95 of 2Wiki's; the path is selective where intermediates are few (median 3, 2 and 1 on MuSiQue, HotpotQA, 2Wiki; the control reaches 10 to 33 percent) and not on MultiHop-RAG (control 98.64) or the hub KBs (MetaQA 55.58, WebQSP 69.71). *Part B:* the target's rank among the source's links, by the pair dot, the question cosine and the residual; question over pair dot at rank 30 is MultiHop-RAG +3.92, MuSiQue +8.32, HotpotQA +6.87, 2Wiki +0.82 (ABOVE), WebQSP +3.00 (LEVEL), MetaQA -25.94 (BELOW): NOT_UNIVERSAL; at rank 10 it is +23.35 on HotpotQA, and on WebQSP -7.92 (ahead only from rank 100, +11.04). *Part C:* sum, z-score and reciprocal-rank fusions are never the worst order (ABOVE the worse on 6 of 6) but fall BELOW the better one on 2 to 3 of 6: NOT_SAFE. Five of six datasets prefer a question-heavy weight; MetaQA alone prefers the pair dot; the label-free correlation of the two scores is lowest on MetaQA (0.377) but does not order the other five, so no weight rule (CR2) is declared.
- **Attack points.** (1) The pool is not read: whether a walk keeps these paths is GP3 A4's, not filed. (2) Rank 30 saturates on short lists (HotpotQA 98.5); the rank-10 read is larger. (3) MetaQA's and WebQSP's slots are answer to answer; the topic-to-answer path, with the relation labels, is what a KB question is, and is not a slot here. (4) The pair dot is the pair's own row dot (GP1's), not a PageRank order.
- **Would change the call.** The question order (or its z75 fusion) lifting pool coverage per distinct node in GP3; a KB read of the topic-to-answer path with the relation labels (MetaQA first; STaRK-Prime held out); more than two KBs telling whether MetaQA's reversal is an outlier or a class.

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
  - R9 part A9 (filed, commit 9c1d873f, four of five datasets): **the cap binds** (candidate recall at 1000/500 over 100/50 is +63.5 MuSiQue, +42.7 MetaQA, +54.8 HotpotQA, +20.0 WebQSP for W0) **but the completion does not follow** (decided without 2Wiki; at 300/150 it is at or below 100/50 on every dataset while the recall gains 12 to 45 points; on the proven subset at 1000/500 it is below on three of four). The 2Wiki cap row and part B9 (the descent) are not read.
- **Attack points.** (1) ROOM says room exists, not that a router reaches it (R8's own words: "H is not a gain"). (2) The hard part is the field: one gold among 229 (MetaQA) to 14,862 (WebQSP) hop-1 nodes; a flat question cosine already reads the question (beats random by 7.7 to 44.6 points), so a branch router has to beat that baseline, and so far does not. (3) R5's gains are on stored links, and four of its six are between 2.5 and 4.2 points. (4) R4 tried one region shape (a trained box is named, not run).
- **Would change the call.** R9 part B9 (the descent) completing clearly more than 60.7 percent or beating the flat question cosine on at least 3 of 5; part A9 has already shown that a lifted cap alone does not.

### 2.7 What ties them (a reading, not a filed result)

1. Where we can see why a gold is missed at the 2,500 pick, it is a cap or budget cut (every gold on MuSiQue and WebQSP, 373 of 375 on MetaQA); on MuSiQue none is ranked out and none unreachable (PL7).
2. The links gold paths need are not unusual: no attribute separates them (GP1, GP2). They do rank early by similarity to their source (above the controls in all nine cells). So the information to find them is mostly closeness, and the lever is **order under a cap**, not a better filter. CR1 adds that ordering by the question beats the pair's own dot on the passage datasets (+0.82 to +8.32 at rank 30) and loses on MetaQA (-25.94).
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
| Chains and link order | CR1: slot distance D/T/N on the whole graph; the rank of the next hop by pair dot, question cosine, residual and fused orders; six datasets read | CR1 ee7ced71 (declared e549c1f4; A1 76d701ab, A3 6c444292, A4 10f8c044) |
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

**Gold coverage by links, uncapped** (UL2H, the hop-coverage ledger). Seeds = dense top 5 plus SPLADE top 5 (about 8 per question). Recall = golds within h links of a seed over all golds, pooled over questions. Nodes = the mean uncapped pool size per question (every link followed, no cap, no ranking), so these are graph-level upper bounds: a cap or a ranking can still lose a gold the graph holds. Unit = the any-link set; version RK for the text corpora, SRK for MetaQA and WebQSP. **2Wiki is not read yet (its host row is running).**

| dataset | golds | seeds only | 1 link: recall (nodes) | 2 links | 3 links |
|---|---:|---:|---|---|---|
| MultiHop-RAG | 2,985 | 71.49 | 99.46 (276) | 100.00 (609) | 100.00 (609) |
| SQuAD | 11,873 | 91.81 | 96.40 (413) | 98.80 (6,512) | 99.87 (17,963) |
| HotpotQA (RK) | 14,810 | 72.80 | 93.54 (280) | 98.89 (1,158,309) | 99.86 (1,164,603 +) |
| MuSiQue | 6,404 | 50.36 | **77.84 [76.67, 78.94] (528)** | 95.64 (24,076) | 99.72 (32,101 +) |
| MetaQA | 72,697 | 0.23 | 13.64 (124) | 64.67 (11,775) | 100.00 (40,663) |
| WebQSP | 10,583 | 2.13 | 72.77 [65.01, 80.76] (11,788) | 98.81 (902,382) | 100.00 (902,382 +) |

(+ = a lower bound: a few questions' balls were unread at three links and are filled by the two-link size.)

**Why a gold is not one link away** (UL2S: the golds the first stage's rrf top five misses, shares of them by class; the classes are exclusive and add to 100; its seeds are the rrf top five, fewer than the table above, so its shares are not that table's complement):

| dataset / version | missed golds | linked to a seed | linked to another gold only (the answer-giving link exists, neither end anchored) | linked, but to no seed and no gold | isolated |
|---|---:|---:|---:|---:|---:|
| MultiHop-RAG | 1,244 | 99.60 | 0.00 | 0.40 | 0.00 |
| SQuAD (one gold per question) | 1,395 | 61.79 | n/a | 38.21 | 0.00 |
| HotpotQA RK / SRK | 4,714 | 77.90 / 87.80 | 7.57 / 8.85 | 14.53 / 3.35 | 0.00 |
| 2Wiki RK / SRK | 11,901 | 88.26 / 91.04 | 5.00 / 5.15 | 6.74 / 3.82 | 0.00 |
| MuSiQue | 3,525 | 59.29 | 18.30 | **22.41** | 0.00 |
| MetaQA | 72,581 | 15.41 | 35.34 | **49.25** | 0.00 |
| WebQSP | 10,424 | 58.43 | 28.71 | 12.86 | 0.00 |

Reading (the two ledgers' pages file only MultiHop-RAG and SQuAD so far; the rest is read from their report files, written 25 Nov, and is an unfiled reading):

- **The links are there.** Within three links every dataset read holds 99.72 percent or more of its golds, and no gold is isolated on any arm. "We do not have the gold links" is not what limits the pools.
- **The price is.** MuSiQue: one link 77.84 percent at 528 nodes, two links 95.64 percent at 24,076 nodes (9.6 times 2,500), three 99.72 percent at 32,101. MetaQA: 13.64, then 64.67 percent at 11,775 nodes, then 100 percent at 40,663 (94 percent of its 43,234-node graph). WebQSP: 72.77 percent at one link already costs 11,788 nodes (4.7 times 2,500; 1,490 new nodes per seed), 98.81 percent at two costs 902,382. The capped pools (PL7) hold 2,500 of these, which is why the misses are cap and order cuts.
- **What the rules do not reach directly.** Between 0.40 and 49.25 percent of the golds the first stage misses have links, but none to a seed or to another gold (MuSiQue 22.41, SQuAD 38.21, MetaQA 49.25, WebQSP 12.86, HotpotQA 14.53 and 2Wiki 6.74 without S; MultiHop-RAG 0.40). Such a gold is reached only through a non-gold row, two or more links out. S helps where it exists (HotpotQA 14.53 to 3.35, 2Wiki 6.74 to 3.82). Which words connect such a gold to the rest of its chain is not measured: the ledger gives the class, not the words, names no new link family, and the 13 Oct word-window read (D4) found no signal on MuSiQue.
- **MuSiQue, by question hops.** The share of missed golds a rule reaches from a seed falls with the question's hops: 71.5 percent on 2-hop, 45.6 on 3-hop, 33.9 on 4-hop questions (UL2S E5). Two figures here land near 78 by coincidence of size: MuSiQue's one-link recall is 77.84 percent (UL2H, seeds = dense five plus SPLADE five) and 77.59 percent of the golds the rrf top five misses are linked to a seed or to another gold (59.29 + 18.30, UL2S; a different seed set and denominator). Neither is the 78 of the oracle read below (SWP Part D); both leave about a fifth outside one link.

### Direct gold links: the oracle result, and what the current links hold

**The oracle read (the user's 78).** SWP Part D (8 Nov, commit 2a733f5; docs/SWP_S_TO_R_SWAP.md sections 5c and 5e) gave the two MuSiQue models of record direct gold links and read in-domain R@5 on 1,000 questions, no new fit. The oracle's chain edges (ORC) join each sub-question's passage to the earlier passages it names ("#1", "#2"): 1,090 distinct pairs. ORA adds every pair of a question's gold passages (1,750 pairs). The two sets were added to the rule graph (R), not swapped for it; the rule graph already held 309 of the 1,090 chain pairs (28.3 percent; read on the 8 Nov graph, which had spaCy NER and a different corpus, so not this page's layout).

| model | rule graph (b1w) | + chain edges (b1c) | + all gold pairs (b1q) | b1c - b1w | decoy partners, b1d - b1w | swapped partners, b1e - b1w |
|---|---:|---:|---:|---|---|---|
| zg6r | 69.0 | **79.6** | 80.3 | +10.7 (+9.6, +11.7) | -1.5 (-2.0, -0.9) | -1.7 (-2.2, -1.2) |
| zrckpn | 67.4 | **78.2** | 78.9 | +10.8 (+9.7, +11.8) | -1.0 (-1.5, -0.5) | -1.0 (-1.5, -0.5) |

By hop depth, b1c - b1w: 2-hop +7.6 / +8.8, 3-hop +15.3 / +13.1, 4-hop +11.5 / +12.6 (zg6r / zrckpn). The fits that never trained on MuSiQue gain more (+16.3 / +15.2). The LLM edges of B1L read 67.4 / 67.3 (no lift over the rule graph). The two controls are the point: edges to a random node (same degree on the golds) and edges to the golds of another question (same gold-to-gold linking, wrong partner) both lower R@5, so the +11 is the connection between the right passages. **It is a ceiling that uses the labels, never an estimate of a buildable graph**, and the oracle edges join gold passages only, so a model that reads "linked to a high-ranked candidate" is helped by construction (the page's own caveat).

**What the current rule links hold against the same kind of direct gold links** (UL2P, the gold-anchored ledger; e3v layout, s1eval; every link of the any-link set, uncapped, whole graph). A slot is two gold passages of one question (gold pairs, GP; on MuSiQue also the chain slots, CH, by the ORC rule). **Recall** = the share of slots whose two ends have a direct link. **Precision** = the share of the links at a gold that end at its partner gold. Links per gold = the mean number of links at one gold. SQuAD has one gold per question, so it has no gold-anchored cell (its seed-anchored read follows).

| dataset | version | slots (questions) | recall, percent | precision, percent | links per gold |
|---|---|---:|---|---|---:|
| MultiHop-RAG | RK | 2,747 (1,130) | 85.55 [83.86, 87.32] | 1.77 [1.70, 1.84] | 89.1 |
| HotpotQA | RK | 7,405 (7,405) | 65.31 [64.32, 66.36] | 0.55 [0.38, 0.75] | 118.0 |
| HotpotQA | SRK | 7,405 (7,405) | 80.01 [79.11, 80.91] | 0.58 [0.39, 0.79] | 138.6 |
| 2Wiki | RK | 26,331 (12,576) | 43.76 [43.26, 44.27] | 1.15 [1.12, 1.17] | 65.6 |
| 2Wiki | SRK | 26,331 (12,576) | 45.80 [45.32, 46.31] | 1.13 [1.10, 1.16] | 69.5 |
| MuSiQue, gold pairs | RK | 5,962 (2,417) | 33.76 [32.58, 34.91] | 0.71 [0.67, 0.75] | 88.7 |
| **MuSiQue, chain slots** | RK | 3,987 (2,417) | **49.76 [48.21, 51.21]** | **0.70 [0.66, 0.74]** | 88.7 |
| MetaQA | SRK | 37,446 (4,160) | 13.12 [12.23, 14.01] | 0.10 [0.09, 0.10] | 603.1 |
| WebQSP | SRK | 4,369 (566) | 19.07 [16.16, 22.25] | 0.01 [0.01, 0.01] | 7,204.8 |
| SQuAD | RK | n/a | n/a | n/a | n/a |

**By label** (the same cells; precision / recall, percent; "-" = no pair or no row; S is the KB triples or hyperlinks):

| dataset | S | R.mention | R.name_comention | R.title_comention | K.knn | any |
|---|---|---|---|---|---|---|
| MultiHop-RAG | - | 0.00 / 0.00 (2 pairs) | 1.77 / 84.89 | 2.08 / 0.29 | 9.96 / 28.14 | 1.77 / 85.55 |
| HotpotQA (SRK) | hyperlink 1.11 / 79.93 | 0.58 / 57.99 | 1.49 / 26.20 | 7.99 / 11.14 | 2.57 / 14.73 | 0.58 / 80.01 |
| 2Wiki (SRK) | hyperlink 2.98 / 42.66 | 1.72 / 40.80 | 1.47 / 22.23 | 6.77 / 3.95 | 3.20 / 9.00 | 1.13 / 45.80 |
| MuSiQue (GP) | - | 0.51 / 18.35 | 1.87 / 22.24 | 3.77 / 1.88 | 2.73 / 7.08 | 0.71 / 33.76 |
| MetaQA | 0.01 / 0.49 | 0.08 / 11.43 | 0.48 / 0.07 | no pairs | 1.36 / 1.79 | 0.10 / 13.12 |
| WebQSP | 0.01 / 6.07 | 0.01 / 16.78 | 1.73 / 0.96 | 0.00 / 0.00 | 1.35 / 2.93 | 0.01 / 19.07 |

**Seed-anchored view** (UL2S: the anchors are the first stage's rrf top five, the targets are the golds it misses, so this is what a walk from the seeds can reach in one link; recall = the share of missed golds with a direct link from a seed, precision = the share of the links at a seed that end at a missed gold, "chance" = the precision of a random link):

| dataset | version | missed golds | recall, percent | precision, percent | chance, percent |
|---|---|---:|---|---|---|
| MultiHop-RAG | RK | 1,244 | 97.83 [96.91, 98.66] | 1.10 [1.04, 1.17] | 0.178 |
| SQuAD | RK | 1,395 | 51.83 [49.27, 54.43] | 0.04 [0.04, 0.05] | 0.001 |
| HotpotQA | RK / SRK | 4,714 | 71.66 / 85.98 | 0.32 / 0.34 | 0.000 |
| 2Wiki | RK / SRK | 11,901 | 86.03 / 90.13 | 0.40 / 0.40 | 0.000 |
| MuSiQue | RK | 3,525 | 49.84 [48.03, 51.66] | 0.29 [0.27, 0.31] | 0.001 |
| MetaQA | SRK | 72,581 | 11.21 [10.65, 11.77] | 1.28 [1.23, 1.34] | 0.017 |
| WebQSP | SRK | 10,424 | 55.63 [49.47, 62.66] | 0.05 [0.04, 0.06] | 0.000 |

Reading (the two ledgers' pages file only MultiHop-RAG, and SQuAD for UL2S; the rest is read from their report files of 25 Nov and is an unfiled reading):

- **The oracle's edges are the slots; ours are a haystack around them.** The oracle gives the model every chain slot at full precision (its edges are the slots). The current layout holds 49.76 percent of MuSiQue's chain slots at 0.70 percent precision: about 1 link in 140 at a gold is its partner. That is far above chance (on the seed-anchored view MuSiQue's precision is 0.29 percent against a chance rate of 0.001), and still a dilution of two orders of magnitude against the oracle's edges.
- **Recall is low where the question is a multi-hop path or a typed KB.** MetaQA 13.12 and WebQSP 19.07 percent of gold pairs have a direct link (S, the KB triples, reach 0.49 and 6.07 of them); MuSiQue 33.76 (49.76 for the chain slots). MultiHop-RAG (85.55) and HotpotQA with its hyperlinks (80.01) are far higher; 2Wiki sits at 43.76 to 45.80. On MuSiQue, UL2's L2e layout connects the whole chain (every chain slot linked) for 72.20 percent of 2-hop, 17.63 of 3-hop and 8.40 of 4-hop questions (s1eval): the longer the chain, the fewer questions have every link.
- **The labels split into wide and precise.** R.mention and R.name_comention carry most of the recall (MuSiQue 18.35 and 22.24 of 33.76; HotpotQA 57.99 of 80.01; WebQSP 16.78 of 19.07; MetaQA 11.43 of 13.12; MultiHop-RAG 84.89 of 85.55) at 0.01 to 1.87 percent precision. R.title_comention and K.knn are the precise ones (1.35 to 9.96 percent where they have pairs) and reach little (0.29 to 28.14 percent; all but MultiHop-RAG's K.knn under 15). A link-level filter has to give up reach to buy precision, which is what UL3, UL4, GP1 and GP2 found (no universal setting; MuSiQue price 86 to 99.5 percent).
- **A flag, not a finding.** The 8 Nov record found the union of S, R, NER and K on 70.1 percent of the 1,090 chain pairs (NER alone 57.7); the current layout (no spaCy) holds 49.76 percent of the 3,987 chain slots. The two are on different graphs, corpora and question sets, so the difference is not a measured loss from removing spaCy. Reading the same 1,090 pairs on the current layout is the check, and has not been run.
- **Not measured: what a pool keeps of these links.** Everything above is the whole graph. How many chain or gold-pair slots have both ends and a link between them inside a 2,500-node pool, and what share of the pool's own links join a slot, is not read; GP3 grades gold coverage (ALL), not the links between golds. GP3 amendment A4 (a530804a, declared before any quarter number) adds that as a beside read, no call and no gate: slot recall in the pool per cell, score and triple from the stored gold coverage, and the precision of the pool's own links on a fixed sample of 40 questions per cell (a re-walk that counts the any-link pairs among the pool's nodes). Its code is not written yet; no number is filed.

**How to restrict, from what was run and what is running.**

- **A per-hop limit.** PL7's per-hop caps and node budgets at 2,500 nodes reach MuSiQue 0.8846, and 0.95 at 6.16 times the bound (section 4's PL7 table). GP3 (declared, amended; running; no coverage number yet) solves per-hop distinct-node budgets N1 >= N2 >= N3 exactly over a 165-triple ladder, with a shuffle null.
- **A score from the question.** The flat question cosine is the reference order: the routed scorer is below it on 4 of 5 (R8 part B), and R9 part A9 finds that the cap binds but the completion does not follow, on the four datasets read (section 6). GP1: on MuSiQue 56 percent of the links gold paths need are in the top 30 of their source's links by question dot (1.8 percent of controls), so the needed links rank high for the question; the lever is the order of the walk, not a filter. GP3 scores K1 to K4 are these question orders.
- **A PageRank order.** GP3 amendment A1 (52301336) adds PP, a PageRank order, beside the question scores. Not read.
- **Link-level removal (UL3, UL4, GP1, GP2).** No universal rule; GP2's final rule keeps 86 to 90 percent of the links, and no setting shrinks MuSiQue.

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
- **Gold links we have (UL2H, UL2S).** Seeds hold 71.49 percent of the golds; one link, 99.46 percent (276 nodes); two links, 100.00 (609 nodes, where the ball stops growing). Of the 1,244 golds the first stage's rrf top five misses, 99.60 percent are linked to a seed and 5 (0.40 percent) only to non-seeds. Nothing here limits the golds a pool can hold.
- **Direct gold links (UL2P).** 85.55 [83.86, 87.32] percent of gold pairs have a direct link, at 1.77 percent precision (89 links per gold); R.name_comention carries it (84.89 at 1.77), K.knn is the precise label (9.96 percent, reach 28.14).
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
- **Gold links we have (UL2H, UL2S).** Seeds hold 91.81 percent of the golds; one link 96.40 (413 nodes); two 98.80 (6,512); three 99.87 (17,963). Of the 1,395 golds the first stage's rrf top five misses, 61.79 percent are linked to a seed and 38.21 percent (533) have links only to non-seed rows; with one gold per question there is no other gold to link to. Why not more: those golds are two or more links from every seed.
- **Direct gold links.** No gold-anchored cell (one gold per question). Seed-anchored (UL2S): 51.83 [49.27, 54.43] percent of the golds the first stage misses have a direct link from a seed, at 0.04 percent precision.
- **What we found.**
  - First stage: e3v is 4.00 points BELOW frozen on R@5 (order, not reach). The windows lift d5 from 83.69 to 89.01 (E6).
  - Links: UL2H shows no gold is isolated: every unreached gold is far, so the lever is ranking and the cap, not a missing link (K.knn at 2 hops reaches 95.19 percent at 77.9 nodes). GP2's final rule keeps 85.5 percent of links.
  - SEM2: R2a adds a little (T2 +0.11); R5's chain beats cosine most here (+7.68 R-precision points); R8: NO_ROOM (cov0 99.60, H 0.33).
- **Issues.** The loss is in the first stage (the encoder), not the links; pools sit close to the line.
- **Open.** Nothing in this phase; the first-stage gap is the vector work that comes after the links.

### 5.3 HotpotQA (S = hyperlinks; shown with and without S)

- **What it is.** The largest source set (about 5.2 million source rows). Two versions, with and without S (the user's ruling 21 Nov). 1,508 dev questions.
- **Where it stands.** ALL 0.9634 at 77.9 nodes.
- **Gold links we have (UL2H, UL2S).** Seeds hold 72.80 percent of the golds; one link 93.54 (280 nodes); two 98.89 (1,158,309 nodes, the uncapped two-link ball); three 99.86 (lower bound; RK). Of the 4,714 first-stage-missed golds: RK 77.90 percent linked to a seed, 7.57 to another gold only, 14.53 to neither; with the hyperlinks (SRK) 87.80, 8.85 and 3.35. Why not more: the hyperlinks add what the text rules do not express; what remains has no hyperlink or rule link to a seed or another gold.
- **Direct gold links (UL2P).** RK: 65.31 [64.32, 66.36] percent of gold pairs at 0.55 percent precision (118 links per gold); with the hyperlinks (SRK) 80.01 [79.11, 80.91] at 0.58, the hyperlinks alone 79.93 at 1.11. Most precise label: R.title_comention 7.99 percent (reach 11.14).
- **What we found.**
  - First stage: e3v is level (-0.43).
  - Links: UL3 does not shrink the RK version (best price 62.2 percent); the SRK version cannot pass under "S never removed" because S holds 52.1 percent of the candidates and 99.9 percent of the covered slots. GP1: 78 to 83 percent of the links gold paths need sit in the top 30 of their source's links, against 0.1 to 0.2 percent of controls; 60 to 68 percent have an exact title match. GP2's cutting rules do cut it (to 6,346 to 8,648 links per question) because a cell's own optimum is only 119 to 193 links per question (HotpotQA RK and 2Wiki RK together), but they are not universal.
  - SEM2: R8 ROOM (cov0 83.82 to 99.87 with at most 1,500 added nodes; 95.1 percent of misses are one hop out); the flat question cosine completes 60.7 percent, the best of any dataset; R4's region is 1.80 BELOW the point; R5 +1.34.
- **Issues.** (1) A cost problem: the kNN groups had to be rebuilt from the record's pointer index (text groups differ in 16 of 5.2 million rows), and the gate row for e3v's K took 85,760 s on the degraded host (about 2 hours when healthy). (2) The cap of 100 branches, not the budget, is the loss on the routing read (92 to 99 percent of candidate recall).
- **Open.** UL2H's own page does not yet file HotpotQA's read (the report is on disk).

### 5.4 2Wiki (S = hyperlinks; shown with and without S)

- **What it is.** Two versions, with and without S. 1,496 dev questions.
- **Where it stands.** ALL 0.9609 at 917 nodes (PL2 needed 2,140; PL3 cut it to 917).
- **Gold links we have (UL2S; UL2H not read yet).** Of the 11,901 golds the first stage's rrf top five misses: RK 88.26 percent linked to a seed, 5.00 to another gold only, 6.74 to neither; with the hyperlinks (SRK) 91.04, 5.15 and 3.82. The hop ledger (how far out, at what price) is the host row still running.
- **Direct gold links (UL2P).** RK: 43.76 [43.26, 44.27] percent of gold pairs at 1.15 percent precision (66 links per gold); SRK 45.80 [45.32, 46.31] at 1.13, the hyperlinks alone 42.66 at 2.98. Most precise label: R.title_comention 6.77 percent (reach 3.95).
- **What we found.**
  - First stage: e3v is +10.85 ABOVE frozen on R@5, the biggest gain of the seven.
  - Links: the one dataset where UL3's screen passes on every cell (cuts 52 to 64 percent of candidates for 2 to 6 points of reach); GP1: 82 to 83 percent of needed links in the top 30 of their source's links against 0.1 percent of controls; 78 to 82 percent exact title. GP2's A2 and P5 rules put it in (the same 119 to 193 links per question own optimum), and no rule of that set puts MultiHop-RAG, MuSiQue, SQuAD or MetaQA in.
  - SEM2: R8 ROOM (cov0 47.66 to 99.93; 98.3 percent of misses one hop out); of all datasets the routed scorer helps only here (+14.50), and a control with another question is level, so the gain is the stored label's, not the question's. R4's region is 2.27 BELOW.
- **Issues.** The slowest row of the phase: R9's 2Wiki cap row takes about 8 minutes per question (75 questions, ends near 03:00 IST). GP1's extraction rows took 1.4 hours.
- **Open.** R9 cap read and UL2H's 2Wiki read (the last host row).

### 5.5 MuSiQue (no S; the hard passage set)

- **What it is.** 117,534 nodes; R and K links only; 2, 3 and 4 hop questions. 1,534 dev, 2,417 eval questions.
- **Where it stands.** **Short of the target at 2,500: ALL 0.8846.** The curve has no cliff: 0.8999 at 3,449 nodes, 0.9222 at 5,742, 0.9350 at 9,076, 0.9429 at 13,615, then **0.9640 at 15,400 (6.16 times 2,500)**. The uncut pool needs 20,038 nodes; **an oracle that picks the depth per question needs 1,831**.
- **Gold links we have (UL2H, UL2S).** Seeds hold 50.36 percent of the golds; **one link 77.84 [76.67, 78.94] (528 nodes)**; two 95.64 (24,076); three 99.72 (32,101, lower bound). Of the 3,525 first-stage-missed golds: 59.29 percent linked to a seed (R, K), 18.30 linked only to another gold (the answer-giving link exists, no anchor), 22.41 linked to neither, none isolated. By question hops the rules' seed reach is 71.5, 45.6 and 33.9 percent on 2-, 3- and 4-hop questions. Why not more: a fifth of the golds sit two or more links from every seed, and the uncapped two-link pool is 24,076 nodes against 2,500, so a pool has to choose among 65.8 new nodes per seed at one link, and far more at two.
- **Direct gold links, the oracle and the current links (SWP Part D, UL2P).** The oracle's 1,090 chain pairs added to the rule graph lifted both models, 69.0 to 79.6 and 67.4 to 78.2 in-domain R@5 (+10.7, +10.8; decoy and swapped partners -1.0 to -1.7). The current links hold **49.76 [48.21, 51.21] percent of the chain slots (33.76 of all gold pairs) at 0.70 percent precision**; whole chains connect for 72.20, 17.63 and 8.40 percent of 2-, 3- and 4-hop questions. How many of these links survive inside a 2,500-node pool is not measured.
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
- **Gold links we have (UL2H, UL2S).** Seeds hold 0.23 percent of the golds (170 of 72,697); one link 13.64 (124 nodes); two 64.67 (11,775); three 100.00 (40,663 nodes, 94 percent of the graph). Of the 72,581 first-stage-missed golds: 15.41 percent linked to a seed, 35.34 to another gold only, 49.25 to neither. Why not more: the golds are answer sets two and three typed hops out, and the seeds hardly touch them.
- **Direct gold links (UL2P).** 13.12 [12.23, 14.01] percent of gold pairs have a direct link, at 0.10 percent precision (603 links per gold); the KB triples (S) reach 0.49 at 0.01, R.mention carries 11.43, K.knn is the precise label (1.36 percent, reach 1.79).
- **What we found.**
  - The first stage finds almost nothing (frozen dense R@5 0.87; GP1: 14 of 3,228 golds); 87 percent of the missed golds are two or three hops out.
  - Misses at 2,500: 33 of 300 questions, 375 golds (11.4 per question), 373 cut by a cap or the hop-3 budget (264 by the hop-3 budget).
  - Links: S is 36 percent of the pairs, but **72.3 percent of the candidates**, and has the lowest one-hop precision (0.005 percent) while being the only unit that reaches ALL 1.0 at three hops. UL2's first KB rule text made F0T 56 times the filed one (706,154 against 12,605 pairs), which broke the size bound until 12.6 rebuilt it from names. UL3 cannot pass under "S never removed"; the cost of that protection is +65 points of price here (UL4: own optimum 17 with S free, 83 with S protected). GP1: MetaQA is the odd one out (its dot, direction and exact-title AUCs lie below 0.5; half of the needed weight is on S).
  - SEM2: R8 ROOM (cov0 18.70 to 85.10 with 1,500 added nodes; the budget costs 29 percent of recall here); R3(c): NO on its 7 relations (all MRR below 2.2, a statement about power, not a finding that it cannot transfer); R4 region 1.02 BELOW; R5 +1.94.
- **Issues.** (1) The golds are typed multi-hop paths the first stage never sees. (2) S is both the lever and the cost, so no rule that protects S can be cheap. (3) The pool needs the three-hop budget.
- **Open.** GP3 (S-only walk reference); UL2H's own page does not yet file MetaQA's read (the report is on disk).

### 5.7 WebQSP (S = KB triples; the explosion)

- **What it is.** A KB graph of 2,592,894 nodes; typed one- and two-hop questions. 1,507 dev, 1,503 eval.
- **Where it stands.** **Short at every size we read.** ALL 0.8789 at 2,095 nodes; 0.8975 at 4,977; 0.9175 at 6,080; 0.9328 at 10,817; 0.9421 at 20,738; the PL6 pick **0.9474 at 39,530 (15.81 times 2,500) does not reach 0.95**. The uncut pool is 39,572 nodes (ALL 0.956); **the oracle depth is 9,301, 3.7 times 2,500**: no depth rule, with or without labels, reaches 2,500.
- **Gold links we have (UL2H, UL2S).** Seeds hold 2.13 percent of the golds; one link 72.77 [65.01, 80.76] (11,788 nodes, 4.7 times 2,500); two 98.81 (902,382); three 100.00. Of the 10,424 first-stage-missed golds: 58.43 percent linked to a seed, 28.71 to another gold only, 12.86 to neither. Why not more: a single link already adds 1,490 new nodes per seed, so the pool has to choose among them, and the first stage's seeds are weak.
- **Direct gold links (UL2P).** 19.07 [16.16, 22.25] percent of gold pairs have a direct link, at 0.01 percent precision (7,205 links per gold; R.mention alone holds 500 million pairs); S reaches 6.07, R.mention 16.78, K.knn is the precise label (1.35 percent, reach 2.93).
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
| GP3 (docs/GP3_NODE_COUNT_RULE.md) | declared 7c85205b; amendment A1 52301336 adds a PageRank order; A2 ac1b882c (WebQSP's walk is capped by construction, at 12 million raw neighbour entries); A3 41a8c45c (cost decisions from HotpotQA's time row: HotpotQA extracts at a 48 million cap, a heavy dataset runs 6 of the quarter's 19 chunks, 2Wiki decided by its own time row); section 13 still "None yet", no coverage number read. A4 a530804a (declared before any quarter number, from the user's request for recall and precision of the links against the gold links): a beside read of slot recall in the pool (gold pairs and MuSiQue chain pairs with a link between them) and of the precision of the pool's own links on a 40-question sample per cell; no call, no gate; its code is next and the old A4 text (the rest-of-dev extraction) is renumbered A5. Quarter rows are running on the host (SQuAD, MetaQA done; MuSiQue, WebQSP shards running; HotpotQA queued for memory) | the least worst-dataset mean distinct nodes at ALL 0.95, exact, with a shuffle null; whether a PageRank or GP2 score order meets 2,500 anywhere it currently does not |
| R9 (SEM2 section 17) | **part A9 filed 9c1d873f** on four of five datasets (CAP_BINDS true, COMPLETION_FOLLOWS false); the 2Wiki cap row is still running (about 340 s per question); part B9 (the descent) not read | whether lifting the 100-branch cap, and then a descent through a question-conditioned tree, finds the thin path the oracle uses |
| UL2H (hop-coverage ledger) | 2Wiki row running (s1eval question 4,259 of 12,576 at 17:28); the other five host rows are on disk (reports of 25 Nov 23:05 to 23:10) and read on this page only, not filed on the ledger's own page | which label or form would link the golds the first stage and walk miss, per dataset |
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

It does not say the 2,500-node target is out of reach for MuSiQue or MetaQA (an oracle depth meets it on MuSiQue at 1,831), nor that WebQSP cannot be reached by a different walk (GP3's order, the S-only walk, and learned question-versus-relation scores are the named routes). It does not say the semantic space is closed (R9 is running). The numbers are development numbers on the current links; the pools are provisional until PL8. The gold-coverage tables in section 4 are uncapped graph readings (a cap or a ranking can still lose a gold the graph holds), read from the UL2H and UL2S report files of 25 Nov for five of the seven datasets (the two ledgers' own pages file only MultiHop-RAG and SQuAD), and 2Wiki's hop ladder is not read yet. They say where a gold sits, not which word or fact links it; that is not measured. The recall and precision tables are the UL2P (gold-anchored) and UL2S (seed-anchored) ledgers read from their report files of 25 Nov (UL2P files MultiHop-RAG only; UL2S files MultiHop-RAG and SQuAD); they are whole-graph readings, not readings inside a pool. The oracle result is of 8 Nov, on an older graph (spaCy NER present) and a different 1,000-question set, and is a ceiling that uses the gold labels.

## 9. Sources

PL7 section 14 (the pool table and the ladders); UL5 sections 11.2 and 11.6 (uncut and oracle depth); UL4 section 10; UL3 section 14; GP1 sections 15.1, 15.7, 15.8; GP2 sections 17 and 18; GP3 section 12; SEM2 sections 7 to 15 (R1 to R7: the operators, the unseen relation, regions, branching, passage and KB, withheld links), 16.12 and 16.14 (R8) and 17 (R9, declared); GP3 sections 12 and 13; ENC E6 (windows); UL2H ledger and report (outputs/ul2p/hops), UL2S ledger and report (outputs/ul2p/seed) and UL2P report (outputs/ul2p), read 26 Nov; SWP sections 5b to 5e (the oracle chain edges, Part D and D2); UL1 (chain connected, defined) and UL2 section 11 (L2e's chain connectivity by hops); ENC section 20.11 and E3W 19.8 (first stage); the project's progress record for the shas of PL2 to PL4, UL2 and the SEM2 rounds.

## 10. Link diagnostics on all ten datasets (22:28 IST, 27 Nov 2026): what is measured, what it says, what is missing

Compiled from the records named in each cell, plus the reads of 27 Nov (GP3's quarter, DRP1's reads, STK3's STaRK reads). **Status labels:** FILED = on a results page of its own; UNFILED = read from a report file; SCREEN = a DRP1 slice (amendment 2: a screen, not the filed read); "none" = not measured. All pools are the current rule pools (not re-tuned); the link set is not declared final (PL8).

| dataset | first stage alone (golds held by the seeds) | one link out: recall (mean nodes) | gold-pair links in the graph: recall / precision | rule pool at its pick: ALL (nodes) | what dense-only without K costs at the 0.95 cell (DRP1) |
|---|---|---|---|---|---|
| MultiHop-RAG | 71.49 | 99.46 (276) | 85.55 / 1.77 | 96.90 (50) | COMPLETE 1,130 q: K 0.00; SPLADE +0.88 [+0.18, +1.59] |
| SQuAD | 91.81 | 96.40 (413) | one gold per question: none; seed-anchored 51.83 / 0.04 | 95.54 (15) | COMPLETE 11,873 q: K 0.00; SPLADE +1.76 [+1.42, +2.11] NEEDED |
| HotpotQA (S = hyperlinks) | 72.80 | 93.54 (280) | 80.01 / 0.58 (RK 65.31 / 0.55) | 96.34 (78) | SCREEN 3,309 of 7,405: K 0.00; SPLADE +2.63 [+2.02, +3.26] NEEDED |
| 2Wiki (S = hyperlinks) | none in the ledger (row unread) | none | 45.80 / 1.13 (RK 43.76 / 1.15) | 96.09 (917) | SCREEN 1,536 of 12,576: K +0.20 [0.00, +0.46]; SPLADE +8.72 [+7.29, +10.22] NEEDED |
| MuSiQue | 50.36 | 77.84 (528); two links 95.64 (24,076) | chain slots 49.76 / 0.70 (pairs 33.76 / 0.71) | 88.46 (2,438); 0.95 needs 15,400 | SCREEN 384 of 2,417: both +1.30 [0.00, +2.86] UNCLEAR (too few); LEG1 A6 dense-only pool -1.3 |
| MetaQA (S = KB triples) | 0.23 | 13.64 (124); two links 64.67 (11,775) | 13.12 / 0.10 | 89.68 (2,336); 0.95 needs 4,973 | not read; LEG1 A6 dense-only pool -1.0 |
| WebQSP (S = KB triples) | 2.13 | 72.77 (11,788) | 19.07 / 0.01 | 87.89 (2,095); 94.74 at 39,530 | COMPLETE 1,503 q: K +0.80 [+0.27, +1.33]; SPLADE +0.53 [-0.13, +1.13]; both +2.00 [+1.20, +2.93] NEEDED |
| STaRK Prime | flat fused list 64.6 at 2,475 nodes | none | none; every single-gold gold has an S link | 85.5 [84.1, 86.7] (2,475; flat 64.6); no cell of 63 reaches 95 (best 91.9 at 5,780) | SCREEN 256 q: K -0.78; both +3.12 UNCLEAR |
| STaRK Amazon | flat fused list 90.5 at 2,500 | none | none | 87.1 [85.4, 88.7] (2,448), 3.3 BELOW the flat list; lifts only at 46 to 220 nodes | SCREEN 746 q: K +0.40 UNCLEAR; SPLADE +3.22 NEEDED; both +3.89 NEEDED |
| STaRK MAG | none | none | none | none (links rows running) | none |

Sources: first-stage and one-link columns = UL2H (UNFILED, 25 Nov, seeds = dense top 5 plus SPLADE top 5) and section 4 above; gold-pair column = UL2P (UNFILED except MultiHop-RAG); pool column = PL7 section 14.2 (seven) and STK3 sections 14 and 20 (FILED, Prime and Amazon); last column = DRP1 (docs/DRP1_DROP_SPLADE_AND_K.md; COMPLETE rows are whole-split reads, SCREEN rows are slices and adopt nothing).

**Extra reads of 27 Nov (all descriptive):**

- **GP3's quarter, solved (nine cells; `outputs/gp3/report_q.md`, not yet filed on GP3's page).** With S kept whole and distinct nodes counted, no per-hop budget order reaches ALL 0.95 in every cell near 2,500, 3,500 or 5,000 nodes (all seven scores infeasible); the best worst-cell mean is 1.0 to 1.3 million nodes; the null (random order) is NOT_BEATING_NULL. The cause is the floor: S's three-hop closure from the seeds alone is 24,638 nodes on MetaQA, 89,221 on WebQSP, 265,506 on 2Wiki and 526,677 on HotpotQA (hyperlinks), before any R link. Own optima: SQuAD 16 nodes, MultiHop-RAG 109, MuSiQue 24,160 (random order 54,115), MetaQA 35,201. WebQSP and 2Wiki are capped by construction. The rule pools at the engine's per-hop caps do better (MuSiQue reaches 0.95 at 15,400), so S has to be budgeted as well, not kept whole.
- **Precision inside a pool (GP3 A4, 40 questions, MultiHop-RAG only so far; `outputs/gp3/multihoprag/precision_q.json`).** The pool's own any-link pairs that join two golds of a question: 0.86 percent at the 25-node floor pool (78 gold pairs, 64 linked, 60 kept; 6,963 pairs), 0.07 percent at 109 nodes (90,277 pairs), 0.006 percent at 609 nodes (1,013,589 pairs). SQuAD, MuSiQue and MetaQA run as host rows `gp3p-*` (queued 22:23); WebQSP, HotpotQA and 2Wiki are not run (their cost rows are hours a question).

**What the ten say, and what they do not.**

1. **Reach is not the limit; the price is.** Within three links every dataset read holds 99.7 percent or more of its golds (UL2H); the first link already costs 11,788 nodes on WebQSP, the second 11,775 on MetaQA and 24,076 on MuSiQue. The links exist, and a pool that follows them all is 5 to 15 times too large.
2. **The links are dilute, not absent.** Precision of gold-pair links is 0.01 to 1.77 percent in the graph (0.006 to 0.86 percent inside a MultiHop-RAG pool) against a chance rate of 0.000 to 0.2 percent: two orders above chance and two to three orders below the oracle's edges (the oracle gives every chain slot at full precision and lifts MuSiQue R@5 by 10.7 points).
3. **K is dispensable wherever it was read.** 0.00 on MultiHop-RAG, SQuAD and the HotpotQA screen; +0.20 on the 2Wiki screen; +0.80 [+0.27, +1.33] on WebQSP (UNCLEAR against the 1.0-point line); -0.78 to +0.40 on the two STaRK screens (UNCLEAR). MuSiQue, MetaQA and MAG are not read. Amazon is the caution: its FULL-minus-STRUCT pairs reach +6.9, and that read does not say whether R or K carries it.
4. **SPLADE is where the dense-only line loses.** NEEDED at the 0.95 cell on SQuAD (+1.76), HotpotQA (screen, +2.63) and 2Wiki (screen, +8.72); smaller on MultiHop-RAG (+0.88) and WebQSP (+0.53 alone, +2.00 with K); NEEDED on the Amazon screen (+3.22). These are the datasets whose questions name their entities, which a lexical leg finds and a dense leg misses: this is the case for testing a question-to-name link as the dense-only line's replacement (not run; to be declared with its own page).
5. **Where the links are wrong, not only dilute:** the golds the first stage misses that link to neither a seed nor another gold are 22.41 percent on MuSiQue, 49.25 percent on MetaQA and 12.86 percent on WebQSP (UL2S, UNFILED), and MetaQA's S has 0.01 percent one-hop precision; these are reached only through a non-gold row two or more links out, so a filter on one-hop labels cannot help them and the order of the walk has to.
6. **STaRK differs in kind.** Prime's walk lifts the flat list by 20.9 points at 2,475 nodes and S carries it (FULL-minus-STRUCT mean -0.59); Amazon's walk lifts only small pools and is level or below the flat list from about 2,000 nodes up; MAG is unread. Prime's missed golds are linked (median union degree 115 against 68 for the found ones), so the loss there is the first stage's rank and the walk's order, not a missing link.

**Missing, in the order they would change a decision:** (a) MAG's links and pools (rows running); (b) a gold-anchored link ledger on the three STaRK datasets (nothing like UL2P exists there; Prime and Amazon have only degree and node-type profiles, STK3 sections 18 and 20); (c) MuSiQue, MetaQA, Prime and Amazon DRP1 reads beyond a few hundred questions (rows running); (d) the in-pool precision read on the six datasets other than MultiHop-RAG (three host rows queued at 22:23 for SQuAD, MuSiQue and MetaQA; three too slow); (e) UL2H and UL2P filed on their own pages (another session's), 2Wiki's UL2H row unread; (f) the dense-only line's name signal (point 4), which is a new link family and needs its own declaration; (g) JNT1 (docs/JNT1_LEARNED_WALK.md, declared 22:25), the learned walk that orders S, R and the seeds' candidates under per-hop budgets, is the other lever on points 1 and 2.
