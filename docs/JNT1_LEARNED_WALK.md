# JNT1: one learned module that walks and ranks, the MLP family first, on subsets (declared 22:25 IST, 27 Nov 2026, before any JNT1 number)

**Nothing here is read yet.** No JNT1 number exists, no model is fitted, no pool is changed, nothing is adopted. This page fixes the question, the arms, the subsets, the gates and the expectations before the first row runs. The code (`outputs/mp_unified/zjnt1.py`, new) is declared here and is not written yet; it gets a self-test before any row runs and is edited afterwards only through an amendment on this page.

## 0. The user's words, and the rulings that bind

- 26 Nov: "we will train in a way that one singular model understands the semantic space and walks along both and can rank simultaneously so it is still only mlp or only gnn / but i am trying to avoid this as much as possible because we need to see how far we can push with the rules itself and then give this as alternative".
- 27 Nov: "yes so instead of having fixed pools we have one module that actually ingests the graphs the same learns the walk depending on the question constructs better pools and ranks accordingly in one single system for mlp and another for gnn".
- 27 Nov: "based on the diagnostics i would prefer if we first drop K and Splade and then later we will check if including it makes any difference, use the SEM2 learnings we have and incorporate them as well for semantic walks too".
- 27 Nov (now): "make sure you use proper subsets and dont waste much time and parallely do the learned model work".
- Binding from earlier: one model per family (one MLP, one GNN; no other model's scores as inputs); the MLP has no message passing; the GNN is its own model; subset first, the full run once per declared rule; every number has a declared rule first; no hub, degree or IDF attribute (24 Nov); every rule on all ten datasets, carried everywhere or nowhere; the GPU is shared, so the MLP step runs on CPU rows.

## 1. The question

The rule pools of the record walk a fixed graph with fixed per-hop caps and a fixed ranking. The lever that survived every link test (UL3, UL4, GP1, GP2, R8) is the order of the links by the question, not their removal; GP3's quarter (solved 27 Nov, `outputs/gp3/report_q.md`) shows that an absolute-budget walk with S kept whole cannot meet a node bound at all (its floor is the closure of S: 24,638 nodes on MetaQA, 89,221 on WebQSP, 265,506 to 526,677 on 2Wiki and HotpotQA with hyperlinks). So the question here is narrow:

> **Does a learned link score, fitted on the train split's gold paths, order the candidates of a budgeted walk (S, R and the first stage's dense seeds only; no K, no SPLADE) better than the parameter-free question-cosine order, at the same node count?**

This is the pool half of the joint module (the walk). Ranking the pool (the other half) is step 2 and is not read here.

## 2. Design of step 1 (the MLP family; no message passing)

- **Graph.** The any-link graph of GP3's `real_context` (the same components, levels and pair dot that the rule pools and GP3 use), **with the K component dropped** (the main line is dense-only, no K; DRP1's `both` arm is its baseline). S stays in the graph and in the features; it is not removed (A.1). **Reading to confirm:** "S never removed" is read as "no S link is deleted from the graph or from the input"; the walk's per-hop width budgets S candidates like the others, ranked by the score (this is how the engine's per-hop caps treat S; GP3's "S whole" is the other reading and is infeasible, see section 1).
- **Seeds.** The dense top 5 of the first stage (e3v), no SPLADE leg. Gate G-J1 checks them against the filed dense lists.
- **Walk.** Three hops. At hop h the candidates are the nodes reached by one link from the nodes kept at hop h-1 (hop 1: from the seeds) that are not yet in the pool; the `w_h` highest-scoring candidates are kept. Pool = seeds + kept nodes; the size is counted as distinct nodes.
- **Scores (arms).**
  - **B0, flat:** the dense list's first n nodes (n = the L arm's mean distinct nodes at the same width triple, rounded).
  - **B1, parameter-free order:** score = cosine of the question and the candidate (e3v), the same widths.
  - **L, learned:** score = a small MLP (one hidden layer of 64, ReLU, logistic output) over the features below, fitted on the train split. One model per dataset for the first read; a pooled fit over datasets is the later generalisation read.
- **Features of a candidate c reached from a kept node p (all label-free; no degree, hub or IDF):** cosine(question, c); cosine(p, c); cosine(question, p); log rank of c by question cosine among the candidates of this hop; log rank of c by question cosine among the out-neighbours of p (the source-relative rank GP1 found the needed links high in); the component of the link (S, W, T0 one-hot; a candidate linked by several components takes the S > W > T0 order of the dedup); the hop (one-hot); the number of distinct kept parents that link to c (the compiled path count the MLP family reads); the best score of any parent (round 1 only). SEM2's operators (rotated vectors, R1) are a named later feature, not in step 1.
- **Labels (train split only).** A candidate is positive if it is a gold node of its question, or lies on a shortest any-link path from the seeds to a gold within the hops that remain. Negatives: up to 200 random candidates per (question, hop), plus all positives. The labels are read only to fit; no gold enters a feature or a walk at read time (the same rule as GP3's probes).
- **Two rounds (the EM-shaped part, training only).** Round 0 fits on the candidates the B1 walk generates. Round 1 re-generates the candidates with the round 0 walk and refits on them (the latent is the path; the E step marks which kept links lie on a gold path, the M step refits the scorer). Round 2 is not declared; a third round needs an amendment.
- **Widths.** Five fixed triples (w1, w2, w3): (10, 20, 30), (20, 60, 120), (40, 150, 400), (80, 400, 1500), (150, 900, 3000), i.e. up to about 65, 205, 595, 1,985 and 4,055 nodes. They are fixed here; no width is added after a number is read. The cap, not the budget, was the loss in SEM2 R8, so the widest triples are deliberately wide.

## 3. Subsets and splits (subset first, the full run only after a declared pass)

- **Fit:** 2,000 train questions per dataset, drawn by a fixed seed from the train split (MultiHop-RAG: all its train questions if fewer). **Read:** the first 300 questions of GP1's dev slice order (`dev_slice`), per dataset. **Never read:** the eval/test split (s1eval) of any dataset.
- **Order of datasets:** MultiHop-RAG on the laptop (its vectors are there; plumbing, all gates, no call); then MuSiQue and MetaQA on the host (the two hard ones: MuSiQue reaches 0.88 at 2,500 nodes, MetaQA 0.90); WebQSP, 2Wiki and HotpotQA only after those two are read (their walks are 100 to 400 seconds a question in GP3's cost rows; the width budgets here should cut that, but the cost is measured on a probe first); the STaRK three need the R store interface of STK2 and follow.
- **Cost rule.** A probe of 20 questions per dataset prices the walk before the fit rows are queued; rows whose estimated time exceeds 4 hours are cut to fewer questions and the cut is written on this page before the row runs.

## 4. What is read, and the calls

Per dataset and width triple: ALL (every gold in the pool), ANY, micro coverage, mean distinct nodes, with paired bootstrap intervals over the 300 questions (2,000 resamples, seed 20261127). The loss/gain unit is points of ALL at equal width.

- **L versus B1** at each width: ABOVE if the lower end of the paired interval is above +1.0 points, BELOW if the upper end is below -1.0, LEVEL otherwise.
- **L versus B0** (the flat list at equal nodes): the same three calls; this says whether the walk is worth having at all.
- **B1 versus B0:** the same calls (the parameter-free walk against the flat list).
- **Step 1 is CARRIED** only if L is ABOVE B1 at three or more of the five widths on **both** MuSiQue and MetaQA; HELD OPEN if it is ABOVE on one of them; NOT_CARRIED otherwise. Carrying adopts nothing: it opens step 2 (the ranking head) and the all-ten read.
- **Reference, not a gate:** the DRP1 `both` arm's ALL at the nearest cell of the rule pools, from `outputs/drp1/` (the rule pool's own number, not recomputed), printed beside each width.

## 5. Gates (a failure stops that dataset's reading; none of its numbers is read)

- **G-J0 (walker):** the walker with every width at 1,000,000 equals a brute-force three-hop BFS ball on toy graphs and on 20 real questions; with the widths of the grid it equals a brute-force top-w selection (self-test, before any real row).
- **G-J1 (seeds):** the dense top 5 equals the filed dense lists' first 5 on the first 50 questions of the read slice.
- **G-J2 (no label in a feature):** a label-shuffle control: the same fit with the positive labels permuted within each question gives an L arm whose ALL is not ABOVE B1 at more than one width (its interval, not a point). If it is, a feature leaks the gold and the dataset's L numbers are not read.
- **G-J3 (B1 stands for the rule family):** at the widest triple the B1 pool's ALL is within 3 points of the any-link uncapped 3-hop reach of the same dataset without K (UL2H's `any` unit, same seeds up to the SPLADE leg), or the difference is written on this page before any call.

## 6. Expectations (written before any number; each can fail)

- **J1.** B1 is ABOVE B0 on MetaQA at three or more widths (its S chains are the point of a walk) and is LEVEL or BELOW on MultiHop-RAG (its first stage already holds 71 percent of its golds and its reach is one hop).
- **J2.** L is ABOVE B1 on MuSiQue at the two middle widths (205 and 595 nodes) by at least 3 points of ALL (60 percent).
- **J3.** L is ABOVE B1 on MetaQA at one or more widths (50 percent): the question cosine is least informative where a relation, not a name, decides the next step, and the component feature carries that.
- **J4.** Round 1 is ABOVE round 0 on at least one dataset at one width (40 percent: IRA's alignment weight went to 0 and its 16 reads were WITHIN; the declared difference here is that the labels are gold paths).
- **J5.** The two features with the largest permutation importance in the fitted scorer are the question cosine of the candidate and its source-relative rank (GP1's finding), on both datasets (60 percent).
- **J6.** G-J2's shuffle control is not ABOVE B1 on either dataset (90 percent).
- **J7.** The step costs at most 2 CPU-hours per dataset for the fit plus the five widths on 300 questions (70 percent); MuSiQue and MetaQA are not above 4.

## 7. What SEM2 hands this step, and what it does not

Carried in: dense top-5 seeds (R2a with the SPLADE leg removed); the source-relative rank and the question cosine as the scores (GP1, R8: the flat question cosine beats the routed scorers on 4 of 5, so it is the baseline B1 and a feature, not a thing to replace); wide per-hop budgets (R8, R9: the cap is the loss); no hub, degree or IDF attribute (hub line closed 24 Nov). **Not** in step 1 and named for later: SEM2 R1's learned rotations as features (they exist for the seven only, none for the STaRK three); R5's branching at chain points; R7's fill of withheld links; R9's descent router; the GNN family (question-gated edges, per-hop beam; needs the card); the ranking head; the SPLADE and K add-back (a measured delta on the same widths and gates once the dense-only line is read); joint training of walk and rank.

## 8. Code, rows and outputs

- `outputs/mp_unified/zjnt1.py` (new, declared here; imports `zgp1`, `zgp3`, `zgp1p` read-only; selftest before any row; edit only through an amendment): commands `selftest`, `probe` (20 questions, prices the walk), `labels` (train-split labels and the round 0 candidates), `fit` (the MLP, one round), `read` (the five widths on the read slice, the arms, the gates, the calls). Records under `outputs/jnt1/<dataset>/` (aggregates committed with `git add -f`; the candidate tables are never committed).
- Rows: laptop for MultiHop-RAG; host CPU rows (`mpr`, no GPU) for the rest, as `python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/host_ops/pylib_run.py outputs/mp_unified/zjnt1.py <cmd> ... --host`, one class name per row, memory asked from the probe's measured peak.
- Disclosure: the plumbing facts above (the loader, the components, the dev slice) were read from `zgp3.py` and `zworld.py` on 27 Nov; no JNT1 or walker number was read.

## 9. Results

None yet.
