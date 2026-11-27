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
