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
