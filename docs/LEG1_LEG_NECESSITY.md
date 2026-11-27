# LEG1: is each leg of the first stage needed, and can the sparse vectors carry more?

**Declared 07:20 IST, 27 Nov 2026, before any number of it.** Owner: the retrieval session. Companion pages: `docs/SPL1_SPARSE_LEG_SCREEN.md` (the sparse leg and its models), `docs/STK1_STARK_PACKAGE.md` (the first stage of the three STaRK sets).

## 1. The question

The user, 27 Nov: "also i want to test another thing is splade even needed and consequently is dense even needed if we have splade, if we just modify the splade vectors would we be able to encode info better?"

Read as two parts:

- **Part A, necessity.** The first stage fuses two lists, e3v dense and SPLADE (C0), by reciprocal-rank fusion (constant 60); the seeds of the walk are the dense top 5 with the SPLADE top 5. Does dropping either leg cost coverage?
- **Part B, modification.** If the sparse vectors are changed by a rule with no training, does one list carry more of what the fusion carries? The encoders stay frozen (the user's ruling of 21 Nov: e3v is the dense encoder; C0 the sparse one).

## 2. Part A: the legs alone against the fusion

**Data.** The record's lists only: the e3v top 1,000 and the C0 top 1,000 of each eval question of a dataset (`zspl1.context`, the same rows, golds and gate G-S1 as `docs/SPL1_SPARSE_LEG_SCREEN.md` section 5). Nothing is encoded. The datasets read by this page: every dataset whose two lists load on the machine that runs the row. SQuAD, MuSiQue and MultiHop-RAG on the laptop first; the others by a host row of the same module, and the verdict is over all datasets read.

**Reads, per question, as set coverage** (distinct golds found over golds, as `reads_of`):

| read | union (the record) | dense alone | SPLADE alone |
|---|---|---|---|
| **seeds** (primary) | dense top 5 with SPLADE top 5, distinct nodes, u nodes | the dense top u | the SPLADE top u |
| R@20 | the fusion's top 20 | the dense top 20 | the SPLADE top 20 |
| R@5, R@100 | the fusion's top 5, 100 | the dense top 5, 100 | the SPLADE top 5, 100 |
| R@1000 | the fusion's top 1,000 (not where n < 1,000) | the dense top 1,000 | the SPLADE top 1,000 |

The budget u is the question's own union size (5 to 10), so a single leg gets exactly as many seeds as the union has: the comparison is at equal budget per question.

**What is lost by dropping a leg** is, per question, the union's coverage minus the other leg's alone (dropping SPLADE leaves dense alone; dropping dense leaves SPLADE alone). Mean and 95% interval over questions by the paired bootstrap of section 5 (1,000 resamples, seed 0), in points.

**Call per leg and dataset, on the seeds read.** NEEDED if the lower end of the loss is at least +1.0 point; DISPENSABLE if the upper end is below +1.0 point; UNCLEAR otherwise. **Verdict per leg over the datasets read:** NEEDED if it is NEEDED on at least one dataset (a universal rule keeps what one dataset needs); DISPENSABLE if it is DISPENSABLE on all; otherwise UNCLEAR. The R@20 read gets the same calls and is reported beside it; it decides nothing.

**What the verdict does.** NEEDED for both legs: the first stage stays as it is, and Part A is closed. A DISPENSABLE call is a first-stage read only: it is followed by a pool read (the gold coverage of the pool at the node counts of the filed pools) before anything is dropped, declared in an amendment before its first number. Dropping a leg also changes the first stage of all ten datasets, so it is the user's decision after that read.

## 3. Part B: modified sparse vectors

**Vectors.** The pooled C0 vectors (the record's pooling, arm l0 of `docs/SPL1_SPARSE_LEG_SCREEN.md` section 8.11, fp32) of the nodes and the eval questions of MultiHop-RAG (609; 2,255) and SQuAD (20,233; 11,873) are stored as sparse matrices by a host row on the CPU (the card is not ours at present). **Gate G-B0:** the top 5 of the stored vectors' dot-product lists equal the record's C0 top 5 for at least 99% of the questions; otherwise the part stops and the lists are not read. The reference for every modification below is the unmodified stored vectors, scored by the same dot product, and its list.

**Modifications without links (B1, B2).** Document cap: keep each node's k_d largest weights, k_d in {128, 64, 32}. Query cap: keep the question's k_q largest weights, k_q in {32, 16, 8}. Read: the SPLADE leg's R@20 and R@1000, and the fusion with the unmodified dense list (R@5, R@20, seeds); the mean number of non-zeros per node and per question. Call against the reference by the bootstrap of section 2: **FREE** if the lower end is at least -1.0 point on every read, **COSTS** otherwise. Purpose: a smaller index and faster scoring; a cap that is FREE is a gain in size, not in recall.

**Modification with links (B3).** Each node's vector is mixed with its neighbours' vectors along the filed links (S, R and K) so that the sparse vector carries the walk's information; the mixing rule, its weight grid, the dev slice and the pass line are fixed in an amendment **LEG1-A1** after reading the loaders of the links and before B0's vectors exist, and no number of B3 is read before it is committed. The question it answers: does a sparse vector that already contains the neighbourhood retrieve golds at later hops in the first stage, with the dense leg alone, the sparse leg alone, or both?

## 4. Expectations, written before any number

- **X1** Dropping the dense leg costs at least 3.0 points of seeds coverage on MultiHop-RAG (dense is NEEDED there).
- **X2** SPLADE alone is within 1.0 point of the union on SQuAD's seeds read (the dense leg is DISPENSABLE on SQuAD).
- **X3** Both legs get the verdict NEEDED over the datasets read (the first stage stays).
- **X4** For B1 and B2, a document cap of 128 and a question cap of 32 are FREE and the caps of 32 and 8 are not.

## 5. Disclosures

- The seeds read is the first stage's coverage, not the pool's. The pool walks up to three hops from the seeds, so a leg that adds little to the seeds can still add to the pool's coverage: this is why a DISPENSABLE call asks for a pool read first.
- Part A reads existing lists; it adds no encode and uses no card. Part B's vectors are made on the CPU, at the cost of time, with the record's pooling; whether CPU and card rounding change any list is what G-B0 checks.
- The legs of the three STaRK sets (Prime, MAG, Amazon) are read when their lists exist; the verdict is over the datasets read, and the page lists which.

## 6. Result of part A on SQuAD, MuSiQue and MultiHop-RAG (07:24 IST; `outputs/leg1/legs_<ds>.json`, `outputs/leg1/partA.json`; module `zleg1.py`, commit 381dcc56)

The reader reproduces the filed first stage (G-S1: SQuAD and MuSiQue equal E0's filed R@5 to 5e-6; MultiHop-RAG's top 5 equal `outputs/mhr1/first/lists.npz` for every question). The seed budget is 7.4 to 7.9 nodes per question. **Seeds coverage (points; the loss is the union minus the other leg alone, with its 95% interval):**

| dataset | union | dense alone | SPLADE alone | dropping SPLADE | dropping dense |
|---|---|---|---|---|---|
| MultiHop-RAG (2,255 questions) | 73.74 | 72.91 | 65.75 | 0.83 [0.04, 1.59] UNCLEAR | **7.99 [7.04, 8.97] NEEDED** |
| MuSiQue (2,417) | 53.00 | 51.81 | 47.30 | 1.19 [0.57, 1.76] UNCLEAR | **5.70 [4.98, 6.40] NEEDED** |
| SQuAD (11,873) | 91.81 | 87.17 | 91.26 | **4.63 [4.16, 5.08] NEEDED** | 0.55 [0.23, 0.86] DISPENSABLE |

**Verdict over the three datasets read: dense NEEDED, SPLADE NEEDED; the first stage stays as it is.** The two legs are needed on different data: dense carries the two multi-hop sets (without it the seeds lose 5.7 to 8.0 points), SPLADE carries SQuAD (without it 4.6 points), and each is nearly dispensable where the other leads (SPLADE adds 0.8 and 1.2 points to the multi-hop seeds, with intervals that straddle the 1.0-point line; dense adds 0.55 to SQuAD). On the R@5 read of MultiHop-RAG the dense list alone (64.29) is above the fusion (61.87): the fusion costs 2.4 points there, which the seeds' union (top 5 of each) hides. Deeper lists lose less: dropping SPLADE costs at most 0.9 points at R@20 and 1.1 at R@100 on MuSiQue and MultiHop-RAG; dropping dense costs 2.2 to 7.4 points at R@20 to R@100 there.

Expectations: **X1** held (dropping dense: 7.99 points on MultiHop-RAG, against the 3.0 written); **X2** held (SPLADE alone 0.55 points below the union on SQuAD); **X3** held, SPLADE only through SQuAD. X4 (part B) is open.

**Not read.** MetaQA, WebQSP, HotpotQA, 2Wiki and the three STaRK sets: their e3v lists are made from the host's vectors (the canonical `retrieval_cache` of these datasets holds the older encoder's lists, which is not the first stage), so they need a host row of the same module. They are not part of this verdict, and they cannot change it: the rule keeps a leg that one dataset needs, and SQuAD shows that for SPLADE and MultiHop-RAG for dense. They would add the size of each leg's worth per dataset, nothing more, and are queued only as a table to fill.

## 7. Amendment LEG1-A1 (07:38 IST): the link-mixed sparse vectors (B3), declared before any number of them

Declared with the stored MultiHop-RAG vectors made but unread, after reading the loader of the links (`zul2p.load_mhr`, import only) and nothing else of the dataset's numbers. MultiHop-RAG's families (609 nodes): R, the rule links `CN_all` (25,005 pairs, mean degree 82); K, the e3v kNN pairs `Ke` (1,366 pairs); S is empty. SQuAD and MuSiQue follow only when their vectors exist (SQuAD's by the host row, MuSiQue's when the card is ours), and then by their own amendment of the graph source.

**Mixing.** The node vector is replaced by the node's own vector plus alpha times the mean of its neighbours' vectors, `v' = v + alpha * mean(v_m for m in N(d))`, over the undirected pairs of the chosen graph; a node with no neighbour is unchanged. The questions' vectors and the dense list are untouched. The vectors are the stored C0 vectors of section 3, the scoring the same dot product.

**Arms** (18): graph in {R, K, R+K} x alpha in {0.25, 0.5, 1.0} x size in {as mixed, then the document cap of 128 of section 3} (the capped arm asks whether the information survives at about the original size).

**Reads** (the reads of `zspl1.reads_of`, per question): the fusion's R@5 and R@20, the seeds (dense top 5 with the leg's top 5), the SPLADE leg's R@20 (R@1000 where n is 1,000 or more); and, for the leg alone, the golds outside the unmodified top 20 that enter the modified top 20 (recovered) and those inside the unmodified top 20 that leave it (lost), per question.

**Selection and call.** The questions are split by row, first half the dev half, second half the held-out half. The arm with the largest mean seeds coverage on the dev half is the pick (ties: the smaller number of non-zeros). Only the pick is called, on the held-out half, by the paired bootstrap of section 2 against the unmodified vectors: **MIXING_GAINS** if the pick is BETTER (lower end above 0) on the seeds and not WORSE (lower end at least -1.0 point) on the fusion's R@5, R@20 and the leg's R@20; **MIXING_LEVEL** if it is not BETTER on the seeds but not WORSE anywhere; **MIXING_COSTS** otherwise. All 18 arms are reported on both halves, labelled as such, and none but the pick is called.

**Expectations, written before any number.** **X5** MultiHop-RAG's pick is MIXING_LEVEL or MIXING_COSTS on the seeds: its fusion lists already hold 91% of the golds at depth 20, and its rule links are dense enough (mean degree 82) that mixing averages in the corpus. **X6** For the SPLADE leg alone, some arm recovers more golds than it loses at depth 20. **X7** The uncapped arms hold at least three times the unmodified number of non-zeros per node.

**Disclosed.** MultiHop-RAG is the cheap test, not the deciding one: it has 609 nodes and few hops to recover. A MIXING_GAINS there is a reason to read MuSiQue (117,534 nodes, the hop-rich set), and a MIXING_COSTS there is no reason to skip it. Nodes with no stored neighbour in a graph keep their vector, so an arm's graph decides how many nodes change. Ties at a score of zero in the lists are broken by node id, the same in every arm.

## 8. Results of part B on MultiHop-RAG (07:43 IST; `outputs/leg1/caps_multihoprag.json`, `outputs/leg1/mix_multihoprag.json`; modules `zleg1b.py` (vectors, host row `leg1b-vmhr`, host sha e78e2b6494b4b27e) and `zleg1c.py` (commit cd581f0e))

**Vectors and gate.** The C0 vectors of the 609 nodes and 2,255 questions were made on the host CPU with the record's pooling. **G-B0 passed with every question's top 5 identical in sequence to the record's** (share 1.0): the CPU vectors reproduce the card's lists on this dataset. Non-zeros: 213.2 per node, 128.1 per question. Reference reads (the unmodified vectors, all 2,255 questions): fusion R@5 61.87, R@20 91.28, seeds 73.74, SPLADE leg R@20 83.92 (MultiHop-RAG has fewer than 1,000 nodes, so there is no R@1000).

**B1 and B2, the caps (differences to the reference, points, with the lower end of the 95% interval; FREE = lower end at least -1.0 on every read):**

| arm | non-zeros | fusion R@5 | fusion R@20 | seeds | leg R@20 | call |
|---|---|---|---|---|---|---|
| document cap 128 | 128 per node (-40%) | +0.61 (+0.20) | +0.19 (-0.08) | +0.17 (-0.05) | -0.19 (-0.50) | **FREE** |
| document cap 64 | 64 | +0.02 | -0.19 | -0.38 (-0.82) | -2.65 (-3.20) | COSTS |
| document cap 32 | 32 | -1.43 | -0.09 | -1.89 (-2.39) | -3.98 (-4.79) | COSTS |
| question cap 32 | 32 per question | -0.78 (-1.32) | +0.64 | -0.19 | -1.06 (-1.59) | COSTS (just) |
| question cap 16 | 16 | -2.04 | -0.75 | -1.03 | -4.38 (-5.12) | COSTS |
| question cap 8 | 8 | -4.84 | -2.28 | -2.13 | -9.80 (-10.77) | COSTS |

A node's vector can be cut from 213 to 128 non-zeros with no loss on this dataset (the fusion's R@5 is even 0.6 points better, BETTER by the lower end). Expectation **X4** held for the document cap of 128 and for the caps of 32 and 8, and **did not hold for the question cap of 32**, which costs 1.1 points of the leg's R@20 (lower end -1.59, past the -1.0 line). The 64 cap, not named in X4, costs 2.7 points of the leg's R@20.

**B3, the link-mixed vectors (LEG1-A1).** Links: R 25,005 pairs, K 1,366, R+K 25,269. All 18 arms were read on both halves (dev 1,127 questions, held-out 1,128); the arm with the largest dev seeds is the pick: **R+K, alpha 0.25, document cap 128** (dev seeds 73.89 against the reference's 73.74). On the held-out half against the unmodified vectors: seeds -0.06 [-0.43, +0.33], fusion R@5 +0.49 [-0.06, +1.07], R@20 +0.31 [-0.02, +0.67], leg R@20 +0.07 [-0.32, +0.44]: **MIXING_LEVEL**, nothing gained and nothing lost. No arm moves the held-out seeds by more than 0.9 points in either direction (range 72.92 to 73.73 against 73.74; the worst is K, alpha 1.0, capped). At depth 20 the leg alone recovers 0.015 to 0.117 golds per question and loses 0.004 to 0.125: the largest recovery (K, alpha 1.0, uncapped, 0.117) comes with the largest loss (0.125), a net of nothing. The mixed vectors without the cap hold 826 non-zeros per node (K) to 5,796 (R+K, 27 times the reference), because the rule links connect every node to about 82 others.

Expectations: **X5** held (MIXING_LEVEL); **X6** held in a trivial sense (several arms recover more golds than they lose at depth 20, by at most 0.05 per question); **X7** held (the uncapped arms hold 3.9 to 27 times the reference's non-zeros).

**What this answers, and what it does not.** On MultiHop-RAG, mixing a node's sparse vector with its neighbours' (rule links or e3v kNN) does not put graph information to use: the first stage holds 91% of the golds at depth 20 already, and the links are too dense on this dataset to select anything. This is the cheap test (609 nodes). It does not read MuSiQue, the hop-rich dataset, whose vectors need the card (117,534 nodes), nor SQuAD, which has one gold per question. The reading of the user's question that remains open is therefore: **whether a graph mixed into the sparse vectors helps where later-hop golds sit far from the question in the first stage**; MuSiQue decides it, by an amendment naming its graph before its vectors are read.

**Disclosed.** The vectors were made on the CPU; G-B0 equal to the record on all questions means no rounding changed a list. Where fewer than 1,000 nodes score above zero, ties at zero are ordered by node id, the same in every arm. The pick was made on the dev half and called on the held-out half; the other 17 arms are not called.

## 9. Amendment LEG1-A2 (10:45 IST): the gate on SQuAD failed as declared; what it showed, and the replacement gate, declared before any SQuAD number of part B

**What happened.** The host row `leg1b-vsq` made C0's vectors of SQuAD's 20,233 nodes and 11,873 questions on the CPU (3 h 01 min of runtime, the nodes 2 h 52 min of it) and stopped at G-B0: the top 5 of the stored vectors equal the record's top 5 **in sequence** for 97.49% of the questions (the gate as written in the code; section 3 said "equal" without saying sequence or set) and **as a set** for 99.05%. Section 3 says that on a failed gate the part stops and no list is read; it stopped, and nothing of part B has been read on SQuAD. (MultiHop-RAG passed with 100%.)

**Diagnosis (no part-B read; the vectors against the record's list only).** 298 of the 11,873 questions differ: 113 in the set, 185 only in order. **No difference is a tie** (0 of the 298 are explained by equal scores). In the 113 set differences, the record's missing node scores within 0.3% (median) and 3.2% (99th percentile) of the stored vectors' own 5th score, so the two sets of vectors differ by small score changes, not by a broken pipeline. The stored vectors' mean non-zeros (195.2 per node, 57.4 per question) equal the card's encode of C0 on SQuAD (A7) to the decimal. The cause is not established; the likeliest is the one section 8.11 already disclosed, that the record's pooling covers the padding positions and so a vector depends on the batch it was encoded in, which the record's own run (batches not known to this page) and the stored run (batches of 64 in file order) do not share on a dataset whose nodes have different lengths; MultiHop-RAG's nodes are all cut at 256 tokens, which is why it matched exactly.

**Replacement gate for SQuAD, G-B0c.** Every comparison of part B is of a modified vector against the **stored unmodified** vector, scored the same way (section 3), so what the gate must show is that the stored unmodified vectors stand for the record's. G-B0c: the stored unmodified vectors' list against the record's C0 list, on the SPLADE leg's R@20 and on the seeds (with the record's e3v list), by the paired bootstrap of section 2: the lower end of the difference at least **-0.5 point on both**. Passed: part B runs on SQuAD as declared (B1 and B2, section 3), with the stored unmodified vectors as the reference; the gate's own reads are filed with the result (`outputs/leg1/gate_gb0c_squad.json`), and the code that makes them is added to `zleg1c.py` after this page is committed (`gate` command; `caps` accepts the stored vectors of a dataset only on a passed G-B0 or G-B0c). Failed: part B stops on SQuAD and is closed there. G-B0 stays as written for MultiHop-RAG (passed). The calls (FREE at -1.0, and the rest) are unchanged.

**Disclosed.** This gate was written after the first one failed and after the diagnosis above, which is why it is declared here, before the SQuAD caps are read, with its line fixed (-0.5, tighter than the calls' -1.0). It reads the stored vectors' list against the record's list on two reads, which is a reading of agreement, not of a modification. The finding also concerns the lever screen of `docs/SPL1_SPARSE_LEG_SCREEN.md` section 8.11: its arm l0 is the same pooling, so the card's l0 lists are expected to differ from the record's by the same small amount on SQuAD, and its reads against the record carry that.

## 10. Results of part B on SQuAD (10:54 IST; `outputs/leg1/gate_gb0c_squad.json`, `outputs/leg1/caps_squad.json`; `zleg1c.py` commit ae17577f)

**Gate G-B0c (section 9) passed.** Against the record's C0 list, the stored unmodified vectors' list reads the same on both cells: the SPLADE leg's R@20 94.500 against 94.500 (stored minus record +0.000, 95% interval [-0.025, +0.025]) and the seeds 91.805 against 91.805 (difference 0, interval [0, 0]); the line was -0.5. The 298 questions whose top 5 differ (section 9) do not move a coverage read. Reference reads (the stored unmodified vectors, all 11,873 questions): fusion R@5 88.24, R@20 95.28, seeds 91.81, SPLADE leg R@20 94.50, R@1000 99.50; 195.2 non-zeros per node and 57.4 per question.

**B1 and B2, the caps (differences to the reference, points; the lower end of the 95% interval in the call column is the worst over the five reads; FREE = at least -1.0):**

| arm | non-zeros | fusion R@5 | fusion R@20 | seeds | leg R@20 | leg R@1000 | worst lower end | call |
|---|---|---|---|---|---|---|---|---|
| document cap 128 | 127.2 per node (-35%) | -0.13 | -0.12 | -0.19 | -0.23 | -0.05 | -0.36 | **FREE** |
| document cap 64 | 64 | -1.72 | -0.96 | -1.64 | -2.86 | -0.60 | -3.19 | COSTS |
| document cap 32 | 32 | -5.81 | -3.43 | -3.84 | -9.48 | -2.26 | -10.02 | COSTS |
| question cap 32 | 32.0 per question (-44%) | +0.05 | -0.03 | -0.14 | -0.03 | +0.01 | -0.24 | **FREE** |
| question cap 16 | 16.0 (-72%) | +0.19 | +0.03 | -0.14 | -0.24 | -0.01 | -0.45 | **FREE** |
| question cap 8 | 8.0 | +0.16 | 0.00 | -0.41 | -1.39 | -0.14 | -1.66 | COSTS |

Expectation **X4** held in full on SQuAD (document cap 128 and question cap 32 FREE; document cap 32 and question cap 8 not). The question cap of 16, not named in X4, is FREE here too. No cap is BETTER by its lower end anywhere; the free caps are LEVEL, with differences of at most 0.24 points in either direction.

**Both datasets together.** The document cap of 128 is FREE on both (MultiHop-RAG 213 to 128 non-zeros, SQuAD 195 to 127; the worst lower ends -0.50 and -0.36), the caps of 64 and 32 cost on both, so a stored document vector of the first-stage SPLADE leg can be cut to its 128 largest weights on the two datasets read. The question caps differ: MultiHop-RAG's questions hold 128 non-zeros on average and a cap of 32 (-75%) costs 1.1 points of the leg's R@20, SQuAD's hold 57 and caps of 32 and 16 are free. The reads fix no cap for questions as a rule of both datasets; a cap that is relative to the question's length was not declared and is not read here. MuSiQue's vectors (117,534 nodes, about 16 hours on the host CPU against about 10 minutes on the card) are not made; a read there waits for the card.

**Disclosed.** The five reads of every arm include the leg's R@1000, which only SQuAD has (MultiHop-RAG has 609 nodes). The 298 questions with a different top 5 (section 9) are in the reference and in every arm alike. The link-mixed vectors (B3) are not read on SQuAD: section 7 declares the graph of MultiHop-RAG only.

## 11. Amendment LEG1-A3 (11:54 IST): caps and link-mixing on MetaQA, the hop-rich test, in place of MuSiQue

**Why.** The user (27 Nov, after section 8's MIXING_LEVEL on MultiHop-RAG) named MetaQA as the better test than MultiHop-RAG for putting a graph into the sparse vectors. Its S links are the knowledge-base triplets themselves, so an answer entity is a graph neighbour of the question's topic entity, and the first stage alone reaches almost none of its golds (the filed e3v first stage on MetaQA, `outputs/enc/e0/e3v/metaqa.json` on the host: dense R@5 0.65%, R@20 1.48%, R@1000 20.28%; fusion R@5 0.45%). MuSiQue (117,534 passages, about 16 hours of host CPU for its vectors) stays unread in part B; the MultiHop-RAG read of section 8 stays filed.

**What is read** (module `zleg1d.py`, host rows; all reads as in sections 3, 7 and 9 unless stated):
- **Vectors.** C0's (arm l0, section 3) of MetaQA's 43,234 nodes and of every eval question, made on the host CPU (the node texts are entity names, short; the card is held by timed rows).
- **Gate.** G-B0c of section 9, directly: the stored unmodified list against the record's C0 list on the leg's R@20 and on the seeds, lower end at least -0.5 point. The share of equal top 5 is recorded as information, not as a gate (section 9 showed that a batch-dependent pooling does not reproduce a list exactly). No first-stage value filed for MetaQA is used as a second gate: the reference of every arm is the stored unmodified vectors read the same way, so a reader offset would be the same in every arm.
- **B1 and B2.** The caps of section 3 unchanged (document 128, 64, 32; question 32, 16, 8); FREE if the lower end is at least -1.0 on every read.
- **B3.** The mixing of section 7, `v' = v + alpha * mean(neighbours)`, alpha 0.25, 0.5, 1.0; sizes full and capped at 128; four graphs from `zul2p.load_host` (imported, edited nowhere): **S** (the knowledge-base triplets), **R** (the rule links, `CN_all`), **K** (the e3v kNN pairs, `Ke`) and **S+R+K** (their union): 4 x 3 x 2 = 24 arms. **Split:** the dev half is the even question index, the held-out half the odd one (MetaQA's rows are grouped by hop count, so the first half would be one hop count; this differs from section 7's first-half split for that reason). The dev half picks the arm with the largest seeds mean (ties: fewer non-zeros, then the name); the held-out half calls the pick against the unmodified vectors by the paired bootstrap, with section 7's rule unchanged (MIXING_GAINS: seeds BETTER by the lower end and no read below -1.0; MIXING_LEVEL; MIXING_COSTS). Reads: fusion R@5 and R@20, seeds, leg R@20 and R@1000.
- **Information only, no call:** per arm, the held-out seeds, the golds recovered and lost at depth 20 per question, and the uncapped non-zeros.

**Expectations, written before any number.** **X8:** the picked arm is MIXING_GAINS on MetaQA (its seeds' lower end above 0), because the answers are S-neighbours of the topic entity and the unmodified first stage reaches under 1% of the golds at R@5. **X9:** the best held-out arm has S in its graph (S or S+R+K), not K alone. **X10:** the unmodified vectors' seeds read below 2% on MetaQA. **X11:** the document cap of 128 is FREE on MetaQA (the names hold few non-zeros).

**Disclosed.** Hubs are not corrected: a node with many neighbours dilutes each neighbour's share to alpha/degree. The graphs are the filed ones; no link is made or changed. Part A (are both legs needed) is not read on MetaQA, WebQSP, HotpotQA or 2Wiki by this amendment; the filed record's first-stage gate for them is on the host and a separate amendment declares it. The arm list is longer than section 7's (24 against 18) because S exists here.

## 12. Amendment LEG1-A4 (12:04 IST): part A on MetaQA with the current SPLADE model, and the cost side of the decision

**Why.** The user (27 Nov) is leaning to drop the SPLADE leg, because its encode takes as long as the dense encode or longer and its loss to recall may be small, and asked for one validation on MetaQA with the current SPLADE model (C0, the record's) before the decision is taken. Section 6 read part A on MultiHop-RAG, MuSiQue and SQuAD only; MetaQA's golds are answer entities far from the question, so its first stage is the hardest case for either leg.

**What is read.** Part A of section 2, unchanged: equal seed budget per question, the loss of dropping one leg by the paired bootstrap, the calls NEEDED (lower end at least +1.0 point), DISPENSABLE (upper end below +1.0) and UNCLEAR, and the depth reads (R@20 and deeper). The dense list is e3v's on the host, the SPLADE list the record's C0 list of the same questions. **Gate G-S1 for MetaQA:** the reader's fusion R@5 equals the filed first-stage record of the host (`outputs/enc/e0/e3v/metaqa.json`, `recall.rrf5`) within 5e-6, the quantity section 2's gate compares on SQuAD and MuSiQue; a failed gate stops the read. The command is `zleg1d.py legs --ds metaqa`, which runs `zleg1.py`'s `cmd_legs` unchanged with its context loader wrapped to supply the filed value (`zleg1.py` is not edited). The four datasets' calls then stand side by side; the verdict rule of section 2 (NEEDED if NEEDED on any dataset) is not re-run by this amendment, and the user takes the decision.

**Expectations, written before any number.** **X12:** dropping SPLADE costs MetaQA less than 1.0 point of seeds coverage (DISPENSABLE or UNCLEAR, not NEEDED), because the unmodified first stage reaches under 1% of the golds at R@5. **X13:** dropping dense costs MetaQA more than dropping SPLADE does.

**The cost side, from records that exist (no new measurement).** On the CPU, with the STaRK package's node texts, the dense pair of e3v encodes at 7.5 nodes/s (gte-modernbert, 1,180 tokens/s) and 24.7 nodes/s (granite-small, 3,866 tokens/s) on 8 threads with other rows running (`docs/STK1_STARK_PACKAGE.md`, the smoke of the dense legs), which is about 904 tokens/s for the pair together; C0 measured 234 real tokens/s on the first Prime nodes at 6 threads, so on that machine the SPLADE leg costs about 3.9 times the dense pair per token. On the card, C0 encodes SQuAD at about 200 nodes/s at the record's settings and at 941.6 nodes/s with the lever l5 (4.31 times; `outputs/spl1/lev/pick.json`, fidelity held), the lever's recall reads still running. **Disclosed:** the two CPU figures were taken under different thread counts and load; no same-text, same-card timing of the dense pair against C0 exists, and none is made by this amendment.

**Disclosed.** Part A reads seeds coverage and the first-stage depth; it does not read what the walk makes of a dense-only seed set, which is the pool read (the target metric) and is not declared here.

## 13. Amendment LEG1-A5 (12:58 IST): the MetaQA question set is the filed record's 9,785, not the package's 407,513; section 12's row stopped at its gate before any number

**What happened.** The part A row of section 12 (`leg1d-lmq`, job 261127-120756) ran 39 minutes and stopped at G-S1, as declared: the reader's fusion R@5 was 0.464 percent against the filed 0.004454 (0.4454 percent). **Nothing was read: no loss number of any leg exists, and X12 and X13 are not read.** The cause is the question set: `zspl1.context` takes every question of the package for a dataset that has no stored dense list in `zspl1.eval_rows` (it keeps the stored list for SQuAD and MuSiQue only), and MetaQA has 407,513 questions, while the filed record (`outputs/enc/e0/e3v/metaqa.json`, `metaqa.npz`) is on 9,785 of them (every fourth test question: rows 329282, 329286, and so on, with the dense top 1,000 of each stored). Section 11's part B row (`leg1d-vmq`, encoding all 407,513 questions on the CPU, about 3 hours) was running on the same wrong set; it was cancelled with 127 of the 398 question shards made (the 407,513 questions in blocks of 1,024), before any document vector and before any number. The wrong set is the package's, section 11's and section 12's text said "407,513" or assumed a few thousand questions; neither is the filed record's.

**What changes.** MetaQA (part A of section 12 and part B of section 11) is read on the filed record's 9,785 questions, the rows of `outputs/enc/e0/e3v/metaqa.npz`, with its stored dense lists. `zleg1d.py` replaces `zspl1.eval_rows` for MetaQA only, for the whole run (vectors, read and legs), by a function that returns the npz's rows and the npz itself, so that `zspl1.context` takes its stored-list path: the dense lists are the record's, the gold counts of the package must equal the record's (`gold_total`, a stop otherwise), and G-S1's value is the record's `recall.rrf5`. G-S1 is unchanged (the reader's fusion R@5 equals the filed value within 5e-6), now on the same 9,785 questions; the question ids of the npz must equal the package's ids of those rows (a stop otherwise). The parity split of section 11 is over these questions (4,893 even index, 4,892 odd). Every other rule of sections 11 and 12 stands: the arms, the caps, the gates G-B0c and G-S1, the calls, the bootstrap, X8 to X13 as written, the decision left to the user.

**Files.** The 127 question shards of the cancelled encode (`q_000000.npz` to `q_000126.npz`, 43 MB, from the 407,513-question order) are renamed on the host to `outputs/leg1/vec/metaqa_all407k_partial` so that `zleg1b.encode_shards`, which keeps any shard that exists, can never take one of them for a shard of the new set; nothing is removed. Part B's vectors, caps and mix are written to `outputs/leg1/vec/metaqa` as section 11 says. The new host rows are `leg1e-lmq` (part A, `zleg1d.py legs --ds metaqa`), `leg1e-vmq` (the stored C0 vectors of the nodes and the 9,785 questions) and `leg1e-rmq` (gate, caps and the 24 arms, after the vectors); the three rows of sections 11 and 12 (`leg1d-lmq`, `leg1d-vmq`, `leg1d-rmq`) are not re-sent.

**Disclosed.** Section 12's cost paragraph stands as written; its "3.9 times the dense pair per token" compares CPU figures taken under different threads and load and is not a per-token cost ratio: by multiply-adds per token, C0 (85M encoder parameters and a 24M-parameter output layer over 30,522 words) is about 0.8 times the dense pair and about 1.3 times one dense model of its size, and the lever made C0 4.31 times faster on the card. A same-text, same-card timing of C0 against the dense pair is not made by this amendment; it is declared separately as amendment A6 before any timing is read. The 407,513 questions of the package are not read on MetaQA by any amendment.

## 14. Results of part A on MetaQA (13:07 IST; `outputs/leg1/legs_metaqa.json`, host row `leg1e-lmq`, job 261127-130419-leg1e-lmq-8391; `zleg1d.py legs --ds metaqa`, commit 9d388c48 on the filed record's 9,785 questions of section 13)

**Gate G-S1 passed:** the reader's fusion R@5 is 0.445 percent, the filed record's 0.004454 (within 5e-6), on the same 9,785 questions; the package's gold counts equal the record's. The row took 108 seconds. The 9,785 questions are one fixed sample, every fourth question of the dev rows (2,498 one-hop, 3,718 two-hop, 3,569 three-hop); no other sample of MetaQA is read.

**Seeds** (the golds covered by the seeds, percent, equal seed budget of 8.22 seeds a question): the union of the two legs 0.694, dense alone 0.908, SPLADE alone 0.360. Dropping SPLADE costs **-0.214** points [-0.317, -0.129] (the dense seeds alone cover more, DISPENSABLE); dropping dense costs **+0.334** points [+0.222, +0.457] (DISPENSABLE). Both calls are far inside the +1.0 line, because the whole first stage reaches under 1 percent of the golds of this dataset at the seeds.

**Depth** (the fusion's R@k against dropping a leg; the loss is the fusion minus the remaining leg, in points, positive = dropping costs; the interval is the paired bootstrap):

| R@k | fusion | dense | SPLADE | dropping SPLADE | dropping dense |
|---|---|---|---|---|---|
| 5 | 0.445 | 0.653 | 0.297 | -0.208 [-0.323, -0.099] DISPENSABLE | +0.148 [+0.071, +0.235] DISPENSABLE |
| 20 | 1.252 | 1.479 | 0.553 | -0.226 [-0.368, -0.087] DISPENSABLE | +0.700 [+0.549, +0.868] DISPENSABLE |
| 100 | 4.045 | 4.180 | 2.157 | -0.135 [-0.349, +0.059] DISPENSABLE | **+1.888 [+1.602, +2.156] NEEDED** |
| 1000 | 27.447 | 20.282 | 29.234 | **+7.166 [+6.610, +7.760] NEEDED** | -1.786 [-2.298, -1.334] DISPENSABLE |

**Expectations.** **X12 held:** dropping SPLADE costs less than 1.0 point of seeds coverage (it costs -0.21, a gain). **X13 held:** dropping dense costs more (+0.33) than dropping SPLADE (-0.21), by 0.55 points.

**What the read shows and does not show.** At the seeds and down to R@20 neither leg is needed on MetaQA, the dense leg alone is the better of the two, and the union at an equal budget is below dense alone. At depth the picture turns: from R@100 the dense leg is NEEDED (+1.9 points), and at R@1000 the SPLADE leg is NEEDED (+7.2 points; SPLADE alone reaches 29.2 percent of the golds against dense's 20.3 and the fusion's 27.4, so on this dataset the fusion at constant 60 is 1.8 points below its SPLADE list alone). Which of the two matters depends on what the pool takes from the first stage: the seeds (dense wins at equal budget) or the deep list (SPLADE wins). Section 2's rule is not re-run here, and **this read is not the pool read**: it says nothing about what the walk makes of dense-only seeds, which is the target metric and is not declared by any amendment. The decision on the leg stays the user's.


## 15. Amendment LEG1-A6 (13:33 IST): the pool read without the SPLADE leg, on MetaQA, MuSiQue, SQuAD and MultiHop-RAG, declared before any number of it

**Why.** The user asked whether the pools are built the same without SPLADE (they are not: SPLADE enters the seeds and the first-stage lists the pool cells start from; every filed pool number of PL2 to PL7 carries it in both places) and then said "yes declare it and run it on those four and finish quick". Section 14 read the seeds and the depth lists; it said nothing about what the walks make of dense-only seeds, which is the target metric. This amendment reads that.

**The rule (one, the same on the four).** Take the filed cells of PL6 (below) on each dataset's eval carve with the frozen first stage, exactly as PL5 and PL6 built them, with one change: **the SPLADE list is replaced by the dense list wherever the pool code reads it.** In the pool code that is (1) the seeds, which become the dense top 5 (five seeds a question instead of the union of the dense top 5 and the SPLADE top 5, about 8.2 on MetaQA); (2) the first-stage lists of the cells: `rrf1000_bN` (the top N of the fusion of the two 1,000-lists) becomes the top N of the dense list, because the fusion of a list with itself keeps its order; `splade_*` cells equal their dense cells; `frozen_union` is the dense top 200. The walks, caps, orders, families, vectors and the grade are PL5's and PL6's, unedited (`zpl2g.py` to `zpl6g.py` are imported, none edited). Seeds and base lists are the only inputs that change; nothing is trained.

**Cells (declared by a rule, not by a look at the dense-only numbers).** For each dataset, PL6's `select.json` names the cells it carried (`needed_names`: the dev R95 and R98 picks, the bounded pick, the frontier picks at each node budget, the older pick); the read takes those **whose filed eval mean node count is at most 16,000** (all four datasets' R95 picks are inside it; the heavier budget picks are left out to finish quickly). The list and the filed numbers beside it are written by `zleg1p.py cells` and committed before any dense-only chunk is read (`outputs/leg1/pool/cells_<ds>.json`).

**Questions and reference.** The eval carve of each dataset (PL5's and PL6's confirmation grids: MultiHop-RAG 1,130, SQuAD 11,873, MuSiQue 2,417, MetaQA 9,785 questions). The reference is the filed eval arrays of the same cells (all-gold flag and pool size per question); no SPLADE-side pool is rebuilt except for the gate. Questions are graded in chunks of 128 in a fixed order (a seeded permutation of the chunk numbers, seed 20261127), each chunk written when it is done, so a read can be made at any time on the chunks that exist and says how many questions it covers (a read on part of the questions is labelled PARTIAL; the final call is made on all of them).

**Gates (before any dense-only chunk is graded).** G-D0: the first 16 questions of the eval carve, graded with the intact SPLADE list by the same code, equal the filed arrays element for element for every listed cell (compared with `gate_ref_<ds>.npz`, cut from the filed grids); a stop otherwise, no dense-only chunk is made. G-D1: the dense-only source's SPLADE list equals its dense list, and the dense-only seeds number five a question.

**Measures (per dataset and cell; dense-only against the filed arrays on the same questions).** ALL (the share of questions with every gold in the pool) with its Wilson 95 percent interval; mean pool nodes; the paired difference of ALL (filed minus dense-only, in points, the loss from dropping SPLADE) and the nodes ratio, by the paired bootstrap of zpl2g (1,000 resamples, seed 20261125). **Call per dataset at its R95 pick** (the cell PL6 carried), by section 2's convention: NEEDED if the lower end of the loss is at least +1.0 point, DISPENSABLE if its upper end is below +1.0, UNCLEAR otherwise. Beside it, decided by nothing: the loss at every listed cell; the **reach table** (per system, among the listed cells, the cell with the smallest mean nodes whose ALL is at least 0.95, and its nodes; a post-hoc read on the eval carve, labelled, not a selection).

**Expectations, written before any number.**
- **XD1 (MetaQA):** the loss at the R95 cell is below 1.0 point (DISPENSABLE or a gain): section 14's seeds read had dense alone covering more than the union.
- **XD2 (SQuAD):** the loss at the R95 cell is at least 1.0 point (NEEDED): section 6's part A had dropping SPLADE cost 4.6 points there.
- **XD3 (MultiHop-RAG):** the loss at the R95 cell is above 0 (the interval's lower end above 0), with the call UNCLEAR or NEEDED: part A had 0.8 points.
- **XD4 (MuSiQue):** the loss at the R95 cell is at least 1.0 point: part A had 1.2 points at the seeds, and the deep walk compounds a lower seed set.
- **XD5:** where the loss is positive the dense-only mean nodes are lower than the filed (five seeds expand less), so that the loss at the same cell is not a loss at equal nodes; the reach table is where equal nodes are read.

**Disclosures.** The cells were chosen on the dev carve with SPLADE in the pool (rule U of PL6), which favours the filed side: a dense-only reselection by rule U on the dev grid is **not** made here and could close part of any gap; the reach table is read on the eval carve. The seed budget changes with the leg (5 against about 8), so the same-cell comparison is not an equal-node comparison. WebQSP, HotpotQA, 2Wiki and the three STaRK graphs are not read. Encode and index cost are not read here (the same-card timing amendment is separate and not declared). This is a read, not a rule change: nothing is adopted, and the decision on the SPLADE leg stays the user's.

**Files.** Module `zleg1p.py` (new; imports `zpl2g.py` to `zpl6g.py` and edits none); `outputs/leg1/pool/<ds>/` (`cells_<ds>.json`, `gate_ref_<ds>.npz` committed; `gate_p<i>.json`, `chunk_<n>.npz` from the host; `read_<ds>.json`, `pool_dense_only.json` from the laptop). Host rows `leg1p-*` (CPU, `--host`, the PL6 rows' runner).


## 16. Results of the pool read without the SPLADE leg (14:47 IST; `outputs/leg1/pool/<ds>/read_<ds>.json`, `outputs/leg1/pool/pool_dense_only.json`; module `zleg1p.py`, commit 637c8562; host rows `leg1p-mt0/mt1` (MetaQA), `leg1p-mu0/mu1/mu2` (MuSiQue), laptop runs for MultiHop-RAG and SQuAD)

All four datasets are COMPLETE (every question of the eval carve: MetaQA 9,785, MuSiQue 2,417, SQuAD 11,873, MultiHop-RAG 1,130). **G-D0 passed on every part** (the intact SPLADE list reproduced the filed arrays of the first 16 questions for every cell, 0 mismatches; MetaQA 7 cells, MuSiQue 8, SQuAD 4, MultiHop-RAG 4), **G-D1 passed** (the dense-only seeds number five a question; the SPLADE list is the dense list). MuSiQue left out two cells above 16,000 filed nodes (`rrf1000_b1600+FULL:h3_bf_v20000~cs-fu` at 20,236 and its `+top15000` at 16,576).

**At the cell PL6 carried (R95), the pool without SPLADE loses 0.9 to 1.8 points of ALL on every dataset** (ALL = the share of questions with every gold in the pool; filed = with SPLADE in the seeds and the first-stage lists; the loss is filed minus dense-only):

| dataset (questions) | R95 cell | filed pool: ALL @ mean nodes | dense-only: ALL @ mean nodes | loss in points [interval] and call | nodes, dense-only over filed |
|---|---|---|---|---|---|
| SQuAD (11,873) | `rrf1000_b15` | 0.9554 @ 15 | 0.9378 @ 15 | +1.76 [+1.42, +2.11] NEEDED | x1.000 |
| MetaQA (9,785) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 @ 4973 | 0.9480 @ 4422 | +1.02 [+0.77, +1.29] UNCLEAR | x0.889 |
| MultiHop-RAG (1,130) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 @ 50 | 0.9602 @ 50 | +0.88 [+0.18, +1.59] UNCLEAR | x1.000 |
| MuSiQue (2,417) | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | 0.9640 @ 15400 | 0.9512 @ 15400 | +1.28 [+0.74, +1.86] UNCLEAR | x1.000 |

**Calls (section 2's convention at the R95 cell): SQuAD NEEDED; MetaQA, MuSiQue and MultiHop-RAG UNCLEAR; no dataset DISPENSABLE.** The three UNCLEAR calls sit on point losses of 0.9 to 1.3 points with intervals that cross the +1.0 line (MetaQA's interval is +0.77 to +1.29, MultiHop-RAG's +0.18 to +1.59, MuSiQue's +0.74 to +1.86), so the data neither put the loss over the line nor under it.

**Equal nodes (the reach table, read on the eval carve among the listed cells; post-hoc, not a selection).** SQuAD: the filed pool reaches 0.95 at 15 nodes (0.9554); the dense-only pool reaches it first at the 50-node cell (0.9694), and the list has no cell between, so what it needs lies between 15 and 50. MetaQA: filed 0.9582 at 4,973 nodes, dense-only first reaches 0.95 at 5,238 nodes (0.9501), 5.3 percent more. MuSiQue: both reach at the 15,400-node cell (0.9640 against 0.9512); no smaller listed cell reaches for either. MultiHop-RAG: both reach at the plain 50-node cell, where the dense top 50 (0.9619) is above the filed fused top 50 (0.9540) by 0.8 points [interval -0.27, +1.77, LEVEL]; the walked 50-node cell with SPLADE seeds is 0.9690.

Every listed cell:

**SQuAD** (loss = filed ALL minus dense-only ALL, points):

| cell | filed | dense-only | loss [interval] and call | nodes ratio |
|---|---|---|---|---|
| `rrf1000_b15` | 0.9554 @ 15 | 0.9378 @ 15 | +1.76 [+1.42, +2.11] NEEDED | x1.000 |
| `rrf1000_b50+KNN:h1_c100+top25` | 0.9818 @ 63 | 0.9705 @ 57 | +1.13 [+0.88, +1.35] UNCLEAR | x0.900 |
| `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 @ 2200 | 0.9982 @ 1572 | +0.09 [+0.03, +0.15] DISPENSABLE | x0.714 |
| `equal_rrf_budget_50` | 0.9798 @ 50 | 0.9694 @ 50 | +1.04 [+0.83, +1.25] UNCLEAR | x1.000 |

**MetaQA** (loss = filed ALL minus dense-only ALL, points):

| cell | filed | dense-only | loss [interval] and call | nodes ratio |
|---|---|---|---|---|
| `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 @ 4973 | 0.9480 @ 4422 | +1.02 [+0.77, +1.29] UNCLEAR | x0.889 |
| `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 @ 2336 | 0.8883 @ 2130 | +0.85 [+0.58, +1.10] UNCLEAR | x0.912 |
| `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 @ 6259 | 0.9501 @ 5238 | +1.13 [+0.88, +1.38] UNCLEAR | x0.837 |
| `rrf1000_b800+STRUCT:h3_c25_v2000` | 0.8999 @ 2699 | 0.8898 @ 2500 | +1.01 [+0.72, +1.32] UNCLEAR | x0.926 |
| `dense_top200+STRUCT:h3_c25_v4000` | 0.9244 @ 3600 | 0.9022 @ 2719 | +2.22 [+1.92, +2.50] NEEDED | x0.755 |
| `rrf1000_b800+STRUCT:h3_c100_v5000` | 0.9599 @ 5602 | 0.9499 @ 5067 | +1.00 [+0.74, +1.28] UNCLEAR | x0.905 |
| `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 @ 2018 | 0.8868 @ 1811 | +0.82 [+0.54, +1.09] UNCLEAR | x0.898 |

**MultiHop-RAG** (loss = filed ALL minus dense-only ALL, points):

| cell | filed | dense-only | loss [interval] and call | nodes ratio |
|---|---|---|---|---|
| `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 @ 50 | 0.9602 @ 50 | +0.88 [+0.18, +1.59] UNCLEAR | x1.000 |
| `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 @ 58 | 0.9611 @ 51 | +1.77 [+0.97, +2.74] UNCLEAR | x0.884 |
| `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 @ 100 | 0.9912 @ 100 | +0.00 [-0.44, +0.44] DISPENSABLE | x1.000 |
| `equal_rrf_budget_50` | 0.9540 @ 50 | 0.9619 @ 50 | -0.80 [-1.77, +0.27] DISPENSABLE | x1.000 |

**MuSiQue** (loss = filed ALL minus dense-only ALL, points):

| cell | filed | dense-only | loss [interval] and call | nodes ratio |
|---|---|---|---|---|
| `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` | 0.9640 @ 15400 | 0.9512 @ 15400 | +1.28 [+0.74, +1.86] UNCLEAR | x1.000 |
| `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 @ 5742 | 0.9082 @ 5350 | +1.41 [+0.74, +2.07] UNCLEAR | x0.932 |
| `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 0.8846 @ 2438 | 0.8589 @ 2449 | +2.57 [+1.78, +3.39] NEEDED | x1.004 |
| `rrf1000_b1600+FULL:h2_bf_v2000~s10-cs-vr` | 0.8904 @ 2950 | 0.8705 @ 2469 | +1.99 [+1.32, +2.65] NEEDED | x0.837 |
| `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` | 0.8945 @ 2980 | 0.8631 @ 2520 | +3.14 [+2.32, +3.97] NEEDED | x0.846 |
| `rrf1000_b1600+FULL:h3_bf_v10000~cs-fu+top7500` | 0.9350 @ 9076 | 0.9268 @ 8500 | +0.83 [+0.17, +1.49] UNCLEAR | x0.937 |
| `rrf1000_b1600+FULL:h2_c100_v20000~s20-cs-rk+top15000` | 0.9429 @ 13615 | 0.9326 @ 13135 | +1.03 [+0.37, +1.74] UNCLEAR | x0.965 |
| `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 @ 2092 | 0.7799 @ 1973 | +2.28 [+1.32, +3.27] NEEDED | x0.943 |

**Expectations (section 15).** **XD1** (MetaQA's loss below 1.0 point) **not held**: +1.02 points, call UNCLEAR, the point just above the line (the cells' losses on MetaQA run +0.82 to +1.13 apart from the dense-top-200 cell at +2.22 NEEDED). **XD2** (SQuAD at least 1.0, NEEDED) **held**: +1.76 [+1.42, +2.11]. **XD3** (MultiHop-RAG above 0, UNCLEAR or NEEDED) **held**: +0.88 [+0.18, +1.59] UNCLEAR. **XD4** (MuSiQue at least 1.0) **held** on the point (+1.28), not on the interval (low end +0.74). **XD5** (dense-only nodes lower where the loss is positive) **held in part**: on the walks whose size is set by their visited budget the dense-only pools are mostly smaller (x0.71 to x0.97 on 15 of the 16 walked cells, one MuSiQue cell x1.004; MetaQA's R95 cell x0.889), and on cells whose size is set by a cap or a list length (the R95 cells of SQuAD, MultiHop-RAG and MuSiQue) the two are equal (x1.000), so the same-cell loss there is a loss at equal nodes.

**What the read shows and does not show.** Dropping the SPLADE leg lowers the pool's all-gold coverage by about one to two points at the filed cells on all four datasets (SQuAD the surest, MultiHop-RAG the smallest and least certain), and costs MetaQA about 5 percent more nodes to hold 0.95. It does not show that the loss is permanent: the cells were chosen on the dev carve with SPLADE in the pool (rule U), and a dense-only reselection by that rule on the dev grid, which could close part of the gap, is **not** made here; the reach table is read on the eval carve; five seeds against about eight is part of the difference at the walked cells. WebQSP, HotpotQA, 2Wiki and the STaRK graphs are not read, and encode cost (section 12's question) is not read here. Nothing is adopted or removed: the SPLADE leg stays in the pipeline, and whether to drop it stays the user's decision.


## 17. Results of part B on MetaQA (14:50 IST; `outputs/leg1/gate_gb0c_metaqa.json`, `outputs/leg1/caps_metaqa.json`, `outputs/leg1/mix_metaqa.json`, `outputs/leg1/vec/metaqa/meta.json`; host rows `leg1e-vmq` (vectors, CPU, 968 s) and `leg1e-rmq` (reads, 32 min); module `zleg1d.py`, commit 9d388c48)

**Gate.** G-B0c (section 9's) **passed**: the stored unmodified vectors' list reads the record's on the leg's R@20 (0.554 against 0.553, difference +0.001 [-0.003, +0.006]) and on the seeds (0.694 against 0.694, difference 0). The share of equal top 5 (information only here, section 11) is 0.955 in sequence and 0.984 as a set, so the CPU vectors do not reproduce every list of the card, as on SQuAD (section 9), and every comparison below is of a modified vector against the stored unmodified vector read the same way. Vectors: 43,234 nodes (40.8 non-zeros each) and 9,785 questions (41.9 each). All values below are percent of golds, differences in points.

**B1 and B2, the caps (all 9,785 questions; reference: fusion R@5 0.449, R@20 1.240, seeds 0.694, leg R@20 0.554, leg R@1000 29.201).**

| arm | doc non-zeros | seeds | fusion R@5 | fusion R@20 | leg R@20 | leg R@1000 (difference [interval]) | call |
|---|---|---|---|---|---|---|---|
| doc128 | 40.8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 [+0.000, +0.000] | FREE |
| doc64 | 39.8 | +0.000 | -0.003 | +0.000 | -0.002 | -0.063 [-0.108, -0.019] | FREE |
| doc32 | 29.4 | +0.010 | -0.000 | +0.011 | -0.012 | -0.616 [-0.763, -0.481] | FREE |
| query32 | 40.8 | +0.000 | +0.000 | -0.002 | -0.022 | +0.150 [+0.049, +0.265] | FREE |
| query16 | 40.8 | +0.001 | -0.036 | -0.047 | -0.073 | +1.389 [+1.079, +1.692] | FREE |
| query8 | 40.8 | -0.004 | +0.021 | +0.157 | +0.009 | +1.617 [+1.150, +2.142] | FREE |

**All six caps are FREE** (no lower end below -1.0 on any read). The document cap of 128 changes nothing at all (the names hold 40.8 non-zeros on average, so no document is cut), the caps of 64 and 32 shrink the stored document to 39.8 and 29.4 non-zeros for a leg R@1000 of -0.06 and -0.62 points. The question caps are free and the tighter ones read **above** the reference at depth: a cap of 16 or 8 non-zeros on the question gives the leg's R@1000 +1.39 [+1.08, +1.69] and +1.62 [+1.15, +2.14] BETTER, while the seeds do not move (+0.001 and -0.004).

**B3, the link-mixed vectors (24 arms; dev half 4,893 questions picks, held-out half 4,892 calls).** The vector of a node becomes `v + alpha * mean(neighbour vectors)` over a graph (S the knowledge-base triplets, 124,669 pairs; R the rule links, 12,605; K the e3v kNN pairs, 96,650; S+R+K their union, 229,774). The dev half's pick is **S, alpha 1.0, capped at 128 non-zeros** (dev seeds 3.347 against the dev reference 0.751). Held-out reads of every cap-128 arm (information; only the pick is called):

| arm (cap 128) | non-zeros per node | held-out seeds (ref 0.637) | fusion R@5 (0.410) | fusion R@20 (1.182) | leg R@20 (0.525) | leg R@1000 (28.965) |
|---|---|---|---|---|---|---|
| S_a0.25 | 111.7 | 0.637 | 0.441 | 1.203 | 0.710 | 32.635 |
| S_a0.5 | 111.7 | 1.195 | 0.429 | 1.447 | 2.044 | 32.178 |
| S_a1.0 (the pick) | 111.7 | 2.813 | 0.414 | 1.851 | 3.647 | 28.066 |
| R_a0.25 | 46.3 | 0.650 | 0.414 | 1.212 | 0.541 | 28.485 |
| R_a0.5 | 46.3 | 0.667 | 0.462 | 1.311 | 0.465 | 27.449 |
| R_a1.0 | 46.3 | 0.701 | 0.488 | 1.274 | 0.516 | 24.848 |
| K_a0.25 | 103.6 | 0.635 | 0.440 | 1.193 | 0.546 | 30.198 |
| K_a0.5 | 103.6 | 0.655 | 0.439 | 1.227 | 0.465 | 30.994 |
| K_a1.0 | 103.6 | 0.625 | 0.461 | 1.308 | 0.458 | 31.636 |
| SRK_a0.25 | 125.7 | 0.637 | 0.414 | 1.118 | 0.463 | 29.045 |
| SRK_a0.5 | 125.7 | 0.632 | 0.398 | 1.079 | 0.463 | 30.011 |
| SRK_a1.0 | 125.7 | 0.633 | 0.347 | 1.078 | 0.430 | 29.231 |

**The pick, called on the held-out half against the unmodified vectors:** seeds **+2.18 [+1.82, +2.57] BETTER** (0.637 to 2.813); the leg's R@20 +3.12 [+2.65, +3.58] BETTER (0.525 to 3.647); fusion R@20 +0.67 [+0.44, +0.91] BETTER (1.182 to 1.851); fusion R@5 +0.00 [-0.06, +0.07] LEVEL (0.410 to 0.414); the leg's R@1000 -0.90 [-2.16, +0.21] WORSE (28.965 to 28.066). **The call as the code makes it (every read other than the seeds counts in "not WORSE", section 11's code `zleg1c.mix_call`): MIXING_COSTS, caused only by the leg's R@1000 (lower end -2.16 below the line of -1.0; its point difference is -0.90 and the interval crosses 0).** Read as section 7's text states it (the not-WORSE reads named there are the fusion's R@5 and R@20 and the leg's R@20), the same numbers would be MIXING_GAINS. The text and the code differ on this one read, the filed call is the code's, and the difference is disclosed, not chosen between. The pick holds 111.7 non-zeros per node against 40.8 (2.7 times), and 43,234 of 43,234 nodes change.

**Expectations (section 11).** **X8** (the pick is MIXING_GAINS, seeds' lower end above 0) **not held as filed** (the call is MIXING_COSTS); the seeds' part of it holds (lower end +1.82). **X9** (the best held-out arm has S in its graph) **held**: S alone, alpha 1.0, in both its full and capped forms, holds the largest held-out seeds (2.813), and alpha 0.5 follows (1.195); R alone, K alone and S+R+K stay between 0.63 and 0.70 against the reference's 0.637 (information; not called). **X10** (the unmodified seeds read below 2%) **held** (0.694). **X11** (the document cap of 128 is FREE) **held, and trivially**: it changes no vector.

**What the read shows and does not show.** On MetaQA a rule with no training, `v + 1.0 * mean(S neighbours)` on C0's document vectors, **lifts the SPLADE leg's R@20 by +3.1 points and the seeds' coverage by +2.2 points on the held-out half, on a base of 0.5 and 0.6 percent**, so the lift is large relative to the base and small in absolute terms: the seeds still reach under 3 percent of the golds. Mixing over R, K or their union moves the held-out seeds by at most +0.06 points (information; not called), which fits the reason this test was chosen (the answer entity is a graph neighbour of the topic entity only in S). It costs 2.7 times the stored non-zeros and reads **no better at fusion R@5** (+0.00) and slightly lower at depth 1,000 (-0.90 [-2.16, +0.22]). It is a first-stage read: **whether the mixed vectors raise the pool's all-gold coverage, the target, is not read** (the pool would be rebuilt with these vectors in the seeds and the first-stage lists); no pool, walk or timing is read here; WebQSP, HotpotQA, 2Wiki and the STaRK graphs are not read; the arms were chosen by the dev half and the call is on the held-out half, but the dev-half pick uses the 24-arm list written before any number. Nothing is adopted.


## 18. Amendment LEG1-A7 (14:57 IST): the cost side measured on one card, the same texts, one after another

**Why.** The user (27 Nov) is leaning to drop the SPLADE leg because its encode "takes the same or even more time than dense encoding", and asked to check how long each takes. Section 12 priced the two on records of different machines and loads (CPU: the dense pair 904 tokens per second, C0 234 tokens per second on its first Prime nodes, hence "about 3.9 times the dense pair per token"; the card: C0 at about 200 nodes per second at the record's settings and 941.6 with the lever l5), and section 12 said in its own Disclosed paragraph that no same-text, same-card timing of the dense pair against C0 exists and that none was made. The ratio between the two legs on the card is the cost side of the decision, and the three STaRK legs have since been run at two different levers on the card shared with other rows, so none of their logged block rates is a clean comparison. This amendment measures it once, with the card held by this row alone.

**What is timed.** Four arms on the same texts, in the paths production runs them (no new encoder code; the new module only calls them):
- **gte** and **granite**: the two parts of e3v, `zenc.encode_texts` through `zstk1d.load_part` (float16 weights, longest first, batches under the part's token budget, the part's own window), as the dense fill makes the blocks.
- **C0 l0**: `zstk1g.Pooler` with lever none (A6's path: float32, batches of 64 in file order padded to the longest, forward passes of 16 rows, 256 tokens).
- **C0 l5**: `zstk1g.Pooler` with lever l5 (A8's path: float16, batches of 64 in file order, forward passes of 32 rows, the next batch tokenized in a second thread; SPL1 A11's pick).

**Texts.** For each of Prime, MAG and Amazon, the first 4,096 rows of docs shards 0 and 3 of the package's `nodes.jsonl` (the text T1 that the dense fills and the legs read; the slices of STK1 A7's gate G-V, there 512 rows), so 8,192 nodes per KB and 24,576 per arm and repeat. The same rows go to all four arms.

**Procedure.** One process, the card held by this row alone (the row asks 0.80 of the card; no other row of this session holds the card while it runs; the gpu guard's yielding to another user is read from the log and, if it happened, the repeat is dropped and said). Each model is loaded once and warmed on 64 rows (not counted). Per KB and arm the wall time of encoding all 8,192 rows (tokenization, host to card, forward, pooling and the copy back; `torch.cuda.synchronize()` at both ends; file reads and model loads outside) is taken in **3 repeats, the order of the four arms rotated by one place each repeat** so that no arm always runs first or last. The row asks 2 CPUs, as the production rows do. Recorded: nodes per second and real tokens per second per arm, KB and repeat, the medians and ranges, the card's compute processes (from `nvidia-smi`) before and after, the non-zeros per node of the two C0 arms.

**Measures.** **Pair rate** = 1 / (1 / gte rate + 1 / granite rate) in nodes per second: the rate at which e3v's two parts together make a node (both parts are needed for one e3v vector). **R_5** = C0 l5 rate / pair rate and **R_0** = C0 l0 rate / pair rate, per KB and over the 24,576 nodes (medians of the three repeats; the range over the repeats beside each). A ratio above 1 means the SPLADE leg makes more nodes per second than the dense pair. No call and no threshold is declared: the ratios are the result, with the three repeats' range beside them.

**Expectations, written before any number.** **T1:** R_5 is above 1 on every KB (with the lever the leg is cheaper than the dense pair on the card). **T2:** R_0 is below 1 on every KB (at the record's settings the leg is dearer than the pair, as the CPU records say). **T3:** R_5 over the three KBs lies between 1.5 and 3.0. **T4 (information, not a gate):** the Prime rates of gte and granite are within 25 percent of G-V's 749 and 1,240 nodes per second (`outputs/stark/dense_gpu/gate_v.json`), which ran with other rows on the card.

**Disclosed.** The dense parts read the whole text up to their window and C0 reads the first 256 tokens, so the legs do different work per node by design, and the ratio is a ratio of the production paths, not of equal work. The pair's two parts are timed one after the other, not overlapped. The first 4,096 rows of two shards are not a random sample: STaRK nodes are ordered by type, so a KB's slices stand for the types they hold, and the three KBs are read separately for that reason. One machine, one card. Not read: the questions' encode, the CPU-side cost of building the SPLADE index or the dense matrix, storage (a node's sparse vector against its 1,152 half-precision numbers), retrieval time. Nothing is adopted or removed by this read; the decision on the SPLADE leg stays the user's.

**Files.** New module `outputs/mp_unified/zleg1t.py` (commands `selftest` and `time`), declared here and committed after this section; it imports `zstk1d` and `zstk1g` and edits neither. Result record `outputs/leg1/timing_card.json`. One host card row, class `leg1t-card`, behind the end of `stk1m-spl3` (so that the card is free), with the dependency on `stk1m-first4` not required.


## 19. Results of LEG1-A7 (15:10 IST): the SPLADE leg against the dense pair on one card (`outputs/leg1/timing_card.json`; host row `leg1t-card`, 379 s, module `zleg1t.py`, commit e1cabe42, host sha `c92fcc333092f89c`)

**Run.** The row ran alone on the card as far as the feeder knows (no other GPU row of this session was running; the two peer rows and `stk1m-first4` are CPU rows); `nvidia-smi` lists 14 processes before and 15 after (one was added during the run), memory not shown under the display driver's mode, so it cannot say that no other process used the card; the row's log has no yield of the gpu guard. Texts: the first 4,096 rows of docs shards 0 and 3 of each KB, 8,192 nodes per KB, text hashes in the record; the same rows in all four arms; three repeats, arm order rotated (`gte, granite, c0_l0, c0_l5`; `granite, c0_l0, c0_l5, gte`; `c0_l0, c0_l5, gte, granite`).

**What the slices hold (read after the run from the package's `type` field, decides nothing):** Prime: shard 0 is 4,096 `gene/protein` nodes and shard 3 is 4,096 `molecular_function` nodes; **MAG: all 8,192 are `author` nodes** (names; 9.7 tokens a node), none of MAG's 700,244 `paper` nodes, which are the type that holds MAG's golds; Amazon: 8,192 `product` nodes. The MAG row therefore stands for author names, not for the KB.

**Rates (nodes per second, median of three repeats; tokens per node is the dense parts' count; the pair rate is 1 / (1 / gte + 1 / granite), range over the repeats beside it).**

| KB (slice) | tokens a node (gte / C0) | gte | granite | pair | C0 l0 | C0 l5 | non-zeros a node (C0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Prime (genes, functions) | 85.7 / 94.4 | 1,277 | 2,530 | 853 [751, 864] | 376 | 1,684 | 96.8 |
| MAG (authors) | 9.7 / 9.2 | 9,973 | 18,811 | 6,518 [6,061, 7,136] | 2,165 | 4,438 | 36.0 |
| Amazon (products) | 268.6 / 230.7 | 428 | 809 | 280 [259, 284] | 216 | 1,066 | 192.5 |
| all 24,576 nodes (total nodes over total seconds) | | 935 | 1,779 | 613 [562, 621] | 387 | 1,714 | |

**Ratios (C0 rate over the pair's rate; median [lowest, highest repeat]).**

| KB | R_5 (C0 with l5 / pair) | R_0 (C0 at l0 / pair) | l5 over l0 (C0's own rates) |
| --- | --- | --- | --- |
| Prime | **1.99** [1.94, 2.24] | 0.44 [0.44, 0.48] | 4.5 |
| MAG (authors) | **0.71** [0.62, 0.72] | 0.34 [0.26, 0.36] | 2.0 |
| Amazon | **3.81** [3.76, 4.09] | 0.77 [0.76, 0.84] | 4.9 |
| all 24,576 | **2.77** [2.76, 3.05] | 0.63 [0.63, 0.67] | 4.4 |

**Expectations.** **T1 (R_5 above 1 on every KB): NOT held** (MAG's authors 0.71, and all three repeats are below 1: 0.72, 0.62, 0.71; Prime and Amazon hold in every repeat). **T2 (R_0 below 1 on every KB): held** (every repeat of every KB is below 1; the highest is 0.84, Amazon). **T3 (pooled R_5 between 1.5 and 3.0): held on the median** (2.77; the first repeat's 3.05 is above 3.0). **T4 (information): not held**: Prime's gte makes 1,277 nodes per second against G-V's 749 (+70 percent) and granite 2,530 against 1,240 (+104 percent); G-V ran with other rows on the card (section 18), so the pair is about 1.7 to 2.0 times faster here than the gate's rates.

**The reading, in the cost's own terms.** With the lever l5 the SPLADE leg makes nodes **2.0 times (Prime) and 3.8 times (Amazon) as fast as the dense pair** on long and medium texts and **0.71 times as fast on author names**; with no lever it is dearer than the pair on all three (0.34 to 0.77 times). The lever is worth 4.5 to 4.9 times on long texts and 2.0 times on short ones. C0 at l5 is about as fast as granite alone on the pooled nodes (1,714 against 1,779) and 1.8 times faster than gte alone (935). **The belief that the SPLADE leg takes the same time or more than the dense encode holds for C0 without the lever and for short texts; it does not hold for C0 with l5 on texts of Prime's or Amazon's length.** Beside it, the pool read of section 16 puts the loss of dropping the leg at +0.9 to +1.8 points of coverage on four datasets (NEEDED on SQuAD only); that number is not part of this section, and nothing is adopted or removed here.

**Disclosed (beside section 18's).** (1) The slices stand for the first types of each KB, and MAG's stand for authors only, so no whole-KB time is computed from them: MAG's papers (700,244 nodes, long abstracts) are the missing read, and a rate for them is not inferred from Amazon's (the leg's own production log, about 1,000 nodes per second over all 1.87 million nodes, mixes the two types). (2) The dense parts read the whole text up to their window and C0 reads the first 256 tokens (Amazon's mean 268.6 against 230.7), so the ratio is of production paths, not of equal work; the dense parts are float16 with longest-first batching, C0 at l0 is float32. (3) The two dense parts are timed one after the other, as section 18 says; run on two cards or overlapped, the pair would be faster. (4) The leg's production block also builds the sparse vectors and merges the top lists, which this timing leaves out. (5) The MAG dense timings are 0.4 to 0.9 seconds per arm, so its rates carry the most noise (the highest repeat is 18 percent above the lowest for the pair). (6) One machine, one card, one day.

**What this section does not show.** The cost of the leg on MAG's papers; the question side's encode; the size on disk (a node's 96.8 or 192.5 non-zeros against the dense pair's 1,152 half-precision numbers); the cost of the first-stage fusion; whether the leg is needed (section 16 is the only read of that, on four datasets). **Nothing is adopted, removed or deleted; the decision on the SPLADE leg stays the user's.**
