# ENC: smaller encoders against the frozen one (D93), stage E0: the first stage

Declared 17 Nov 2026, 04:15 IST, before any number. Code: `outputs/mp_unified/zenc.py`. Records: `outputs/enc/`.
E1a (section 9) declared 17 Nov 2026, before any E1 number. Code: `outputs/mp_unified/zenc1.py`.

## 1. Why

The user, 5 Nov 2026: "i would want to reopen the encoder part and compare against smaller encoders like
Qwen3-Embedding-0.6B, MPNet, or encoders much more suited for the task of this semantic space we are planning to create
here, because if links can carry a brunt of the ranking and reranking signal then we should be able to scale down the
encoders without much loss". On 16 Nov the user held MultiHop-RAG's encoding (D84) until the encoder is fixed. On
17 Nov the user said to run this stage beside D91 ("do both if host has space"), which is also the go for the downloads.

Every number so far uses the frozen substrate: gte-Qwen2-1.5B-instruct, 1536 dimensions, RECORD_SHA256 58958f33
(the CRAG freeze). This stage leaves that substrate as it is. A new encoder is a new substrate under `outputs/enc/<enc>/`,
with the model revision pinned in `outputs/enc/models.json`, and it is never mixed with the frozen vectors.

The hypothesis is the user's: a smaller encoder costs something at the first stage, and the links and models win most
of it back. So ENC runs in stages. E0, this page's numbers, measures the first-stage cost. E1 (graph level), E2 (one
screen fit per encoder) and E3 (three seeds for the chosen encoder) each get their rule on this page before their numbers.

## 2. The arms

| arm | model | dim | params | query input | document input | length |
|---|---|---|---|---|---|---|
| `frozen` | Alibaba-NLP/gte-Qwen2-1.5B-instruct (the record's vectors, read in place) | 1536 | 1.5B | the record's instruction | text verbatim | no cut (32,768) |
| `qwen3e06` | Qwen/Qwen3-Embedding-0.6B | 1024 | 0.6B | its own `query` prompt (the same web-search instruction) | text verbatim | its config's default |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 | 768 | 110M | none | text verbatim | its config's default (384) |
| `mpnetqa` | sentence-transformers/multi-qa-mpnet-base-cos-v1 (MPNet tuned for question-to-passage search) | 768 | 110M | none | text verbatim | its config's default (512) |

**The text contract is the record's.** Document row i is line i of `nodes.jsonl`, the `text` field verbatim: no title and no
prefix (KB nodes are names only, regime 1). Query row j is query_ids.json[j], the j-th line of the split files in the order
the queries manifest names. Each model uses its own published query prompt and no document prompt. Vectors are L2-normalised,
stored as float16, in the record's shard layout (40,000 rows a shard). Rows longer than a model's limit are cut by its
tokenizer, and the cut rows are counted per shard and reported. All encoding runs on the host GPU in fp16, as the record's
GPU rows were.

## 3. Gates (each refuses the next step until it passes)

- **G0 (the code reproduces the record).** On the laptop, `zenc.py g0` encodes 32 document rows and 32 query rows per dataset
  (rng seed 0 over all rows) with the frozen model through the same code path the new encoders use. Every row's cosine
  with the record's stored row must be at least 0.99. This checks the text field, row order, query order and the query
  prompt. `encode` refuses without `outputs/enc/g0.json` = PASS.
- **G0b (the search and the first stage reproduce the record).** On the host, for the frozen arm on each dataset's s1eval
  questions:
  - (i) The scores zenc computes for the record cache's listed ids (`retrieval_cache/dense_top1000.npz`) match the cached
    float16 scores within 0.002 (max abs). The record's cache came from a TF32 GEMM, so ranks may differ at near-ties.
    Also, the cache's top 5 are inside zenc's top 10, and zenc's top 5 are inside the cache's top 10, on at least 99.5%
    of questions.
  - (ii) RRF over the record cache's two lists reproduces the record reads' `rrf` reference: golds in the top 5 are equal
    on at least 99.5% of questions.
  - (iii) Reported, not gated: RRF R@5 with zenc's own dense list against the record's.
- **G0c (integrity).** Every new row is finite, not all zero, with |norm - 1| at most 0.01 after the float16 cast, as the
  record's `dense_integrity` defines a damaged row. Any damaged row stops the encoding.
- **Substrate pin.** Every step reads the mirror's `CANONICAL_FREEZE.json` and refuses unless RECORD_SHA256 is 58958f33...

## 4. E0: the first stage (this page's numbers)

**Questions.** Each dataset's s1eval questions: the ids and gold counts of the record reads
(`outputs/screen/fits/scr-gnx-nx-s0/reads/<ds>__s1eval.npz`). The six datasets hold 45,559 questions: squad 11,873,
musique 2,417, hotpotqa 7,405, 2wiki 12,576, metaqa 9,785, webqsp 1,503. Golds are `gold_node_ids`, resolved by the node
id to position map of `nodes.jsonl`. A dataset whose resolved gold count differs from the read's `gold_total` on any
question stops.

**Search.** For every arm the search is the same: an exact float32 inner product of the query row against every document
row of the full corpus, keeping the top 1,000. Ties go to the lower node id, as the record's pools order them. The frozen
arm reads the record's vectors in place, so nothing is copied.

**Columns** (per question, over questions with a gold; recall = golds found / golds):

- dense R@5, dense R@20, dense R@1000 (the reach a pool can start from);
- RRF R@5: the first stage the models start from. It is 1/(60 + dense rank) + 1/(60 + SPLADE rank) in float32 over the
  two top-1000 lists, with the record's SPLADE list unchanged, ordered by score and then by node id.

**Intervals and calls.** For each new arm against the frozen arm, the paired difference per question, with
`zap1.boot_diff` (2,000 resamples of the questions, seed 0, one slot per question). The call is ABOVE, LEVEL or BELOW at
95%. Cost columns: encoder parameters; vector bytes per row (dim x 2); host GPU seconds per million documents; and,
on the laptop CPU for one query at a time, query encode milliseconds (median of 200 queries, pinned to LP 2 as in
FSV's timing rule).

**What E0 decides.** Nothing is adopted at E0. It gives the first-stage price of each encoder. Three expectations, each
called HELD or NOT_HELD:

- **X1.** Each smaller encoder is BELOW the frozen arm on dense R@5 on 2wiki, hotpotqa and musique (the premise: the first
  stage pays).
- **X2.** On each passage dataset (squad, hotpotqa, 2wiki, musique), the RRF R@5 gap (new - frozen) is closer to zero
  than the dense R@5 gap: SPLADE carries part of the loss.
- **X3.** On each passage dataset, the dense R@1000 gap is closer to zero than the dense R@5 gap: pools can still reach
  what the top 5 lose.

**D94** (open; default: all three new arms go to E1). Only an arm that fails G0c, or whose RRF R@5 is BELOW the frozen
arm by more than 10 points on three or more of the six datasets, stays out of E1. Any other exclusion is the user's call.

## 5. What follows (each declared on this page before its numbers)

- **E1, graph level.** For each carried encoder: the K family rebuilt from its vectors under the record's kNN rule (the
  package's 3 nearest neighbours), the first stage's pools under the record's pool rule, then gold coverage and
  oracle-link reach against the frozen substrate. This is the user's rule: check the links before any refit.
  E1 runs in two parts:
  - **E1a** (section 9): each arm's K at UL1's oracle pairs. It needs only the neighbours of the gold ends, so it runs on
    the CPU as soon as an arm's vectors exist.
  - **E1b**: the pools and gold coverage. It needs each arm's full K over the corpus and the PKO chain's substitutions.
    It is named here and declared on this page after E0 and E1a, before its numbers.
- **E2.** One seed-0 screen fit each of zrckpn and GNX per carried encoder, on D83's layout, read on s1eval.
- **E3.** Three seeds for the encoder E2 selects, then D93's call (which encoder MultiHop-RAG is encoded with: D84).

## 6. Rows, order and cost (estimates; the host's measured numbers replace them)

1. `enc-models` (CPU, now): download and pin the three new models (about 2.9 GB), `outputs/enc/models.json`.
2. G0 on the laptop (CPU, minutes): the laptop holds the frozen model and the CRAG package.
3. `enc-first-frozen-<ds>` (CPU, now, 8 threads): G0b and the frozen arm's E0 rows, about 2 to 8 minutes a dataset.
4. `enc-encode-<enc>-<ds>` (GPU, after D83's last GPU fit, one at a time). mpnet and mpnetqa come first (about 1.5 to 2 h
   each for 14.0M documents and 0.88M queries), then qwen3e06 (about 8 to 10 h). Each row resumes by shard.
5. `enc-first-<enc>-<ds>` (CPU) after each encode, then `enc-report`.

**Disk.** New vectors use about 28.7 GB (qwen3e06), 21.5 GB (mpnet) and 21.5 GB (mpnetqa), plus about 3 GB of queries.
At declaration the host had 152 GB free; the 30 GB floor is kept. The search keeps only the s1eval top-1000 ids and
scores (about 0.4 GB in all). The vectors of an encoder E1 does not carry go on a delete list at E1's filing; the delete
itself is the user's.

## 7. What it can and cannot show

E0 can show how much each smaller encoder loses at the first stage, at what cost saving, and whether SPLADE and the pool
reach cover part of the loss. It cannot show the hypothesis itself (links and models win the loss back): that is E2's
question. No ENC number goes into a paper sentence before its section below is filed.

## 8. Results

**Scope (18 Nov 2026).** The user held the big datasets: "check only the small and hard datasets and see what do we
lose and can we gain back". The nine `enc-encode-<enc>-<hotpotqa|2wiki|webqsp>` rows wait on `hold-enc-big`, and no
item has that name. E0 is read on SQuAD, MuSiQue and MetaQA by `enc-report-small`, which is the same command as
`enc-report`; its record is `outputs/enc/e0_report.md` and `.json` on the host. The declared expectations X1 to X3 and
D94 need the six datasets. They stay **not read**, and the readings below describe the small datasets only. G0b passed
on all six datasets.

**E0, RRF R@5 and dense R@5 (new minus frozen, points, 95%):**

| arm | SQuAD dense R@5 | SQuAD RRF R@5 | MuSiQue dense R@5 | MuSiQue dense R@1000 | MuSiQue RRF R@5 | MetaQA RRF R@5 |
|---|---|---|---|---|---|---|
| frozen (level) | 87.69 | 90.53 | 45.72 | 87.25 | 47.34 | 0.47 |
| qwen3e06 | -6.33 BELOW | -3.36 BELOW | -4.46 BELOW | -3.65 BELOW | -2.36 [-3.04, -1.63] BELOW | -0.12 BELOW |
| mpnet | -14.18 BELOW | -6.55 BELOW | -11.89 BELOW | -7.29 BELOW | -6.10 [-6.90, -5.27] BELOW | +0.07 LEVEL |
| mpnetqa | -11.17 BELOW | -5.52 BELOW | -11.32 BELOW | -7.45 BELOW | -6.42 [-7.23, -5.57] BELOW | -0.02 LEVEL |

- **The small-dataset reading of X2 and X3.** On SQuAD and MuSiQue, for every arm, the RRF gap and the dense R@1000
  gap are both closer to zero than the dense R@5 gap. SPLADE and the pool reach carry part of the loss.
- **The small-dataset reading of X1.** On MuSiQue, every arm is BELOW on dense R@5.
- **MetaQA.** MetaQA's golds are answer sets, and every arm's RRF R@5 there is under 1%.
- **Cost.** Each arm's cost, against the frozen gte-Qwen2-1.5B:

  | arm | parameters | bytes per row | host GPU seconds per million documents |
  |---|---|---|---|
  | qwen3e06 | 596M | 2,048 | 3,173 |
  | mpnet | 109M | 1,536 | 479 |
  | mpnetqa | 109M | 1,536 | 490 |

**E1a** (section 9). Read on MuSiQue and MetaQA by `enc-e1-report-small`; its record is `outputs/enc/e1/report.md` and
`.json` on the host. HotpotQA, 2Wiki and WebQSP were not graded. Shares are % of slots, and differences are points
against the frozen arm, rebuilt by the same code.

| set (frozen K alone / union) | arm | K alone - frozen | union - frozen |
|---|---|---|---|
| MuSiQue GP, RK (7.28 / 33.63) | qwen3e06 | -1.36 [-1.89, -0.87] BELOW | +0.18 [+0.02, +0.36] ABOVE |
| | mpnet | -1.59 BELOW | +0.13 LEVEL |
| | mpnetqa | -2.08 BELOW | -0.10 LEVEL |
| MuSiQue CH, RK (10.89 / 49.71) | qwen3e06 | -2.03 BELOW | +0.28 [+0.03, +0.53] ABOVE |
| | mpnet | -2.43 BELOW | +0.15 LEVEL |
| | mpnetqa | -3.16 BELOW | -0.20 LEVEL |
| MetaQA GP, SRK (2.48 / 13.81) | qwen3e06 | -0.22 BELOW | -0.12 LEVEL |
| | mpnet | -0.05 LEVEL | -0.04 LEVEL |
| | mpnetqa | -0.58 BELOW | -0.57 [-0.72, -0.44] BELOW |

- **G2a reproduced UL1's filed F0T shares** on both datasets.
- **Y1 to Y3, read on MuSiQue only.** They hold for all three arms there. Each arm's K-alone share is BELOW, and the
  union gap is closer to zero than the K-alone gap, on both GP and CH. The declared calls need 2Wiki and HotpotQA, so
  they stay **not read**.
- **D97** stays open.

**What the small datasets say.** On the hard multi-hop set (MuSiQue), what a smaller encoder loses is in the first
stage: 2.4 to 6.4 points of RRF R@5. It loses almost nothing in links. S and R hold the union level or above, because K
is a small part of it. MetaQA's one BELOW (mpnetqa's union, -0.57) comes from K: 2.32 of the frozen union's 13.81 points are slots that only K links. The
place to gain the loss back is therefore the first stage. SEM1 (`docs/SEM1_GRAPH_SPACE.md`) reads one parameter-free
way to do that on these datasets. Nothing here is adopted.

## 9. E1a: each encoder's K at the oracle pairs (declared 17 Nov 2026, before any E1 number)

Code: `outputs/mp_unified/zenc1.py`, new. It imports `zenc.py` and `zul1.py` and edits neither. Outputs go only under
`outputs/enc/e1/`, and its arrays stay uncommitted.

**What E1a does not touch:**
- Nothing trains and no model is read.
- No filed graph, pool, fit or record changes.
- It reads no E0 number. It needs only each arm's document vectors.

### 9.1 Why

K is the only link family an encoder changes. S, R's rule links (W2_S) and F0T are text rules, the same for every arm.
The user's rule (16 Nov) is to check links against the oracle links before any refit. So E1a asks two questions:
- Does a smaller encoder's K still link the oracle pairs as the record's K does?
- How much of any loss do the encoder-free families already cover?

### 9.2 The K rule (the record's)

The rule comes from GRAPH_MANIFEST's `source_manifest` and from `CRAG/scratchpad/knn_parity_check.py`:
- The score is an exact float32 inner product of L2-renormalised float32 copies of the stored float16 rows.
- The search runs over distinct source rows: the top 4, including the row itself. Self is dropped; when self is not among
  the 4, all 4 are kept.
- Each position inherits its source row's neighbours, mapped to the lowest position of each neighbour. The family is
  undirected.

E1a applies this rule in three ways:
- **Source rows.** The package carries no pointer index, on the laptop or on the host. So E1a stands in for a source row
  with a text group: the positions whose `text` is byte-identical. The group's vector is its lowest position's row. G1a
  checks this stand-in on the frozen arm.
- **Only the slot ends.** E1a computes neighbour lists only for the groups of the slot ends, which are the golds of
  questions that have a slot. Each list is the top 5 by score, then by group. The 5th entry marks ties: a row is a tie
  row when its 4th and 5th scores are within 0.001, the record's `tie_tolerance`.
- **Exact ties.** Ties go to the lower group. The record's tie order at the 4th place was arbitrary, so G1a counts a
  difference there as tie churn.

A slot (a, b) is linked by an arm's K when b is in out(a) or a is in out(b). Here out(x) is the set of lowest positions
of the neighbours of x's group. Both ends of every slot are slot ends, so this membership is exact.

The same code runs for every arm, including `frozen`, which is rebuilt from the record's vectors.

### 9.3 Questions and slots (UL1's, unchanged)

- **The questions:** each dataset's s1eval carve (U2b's questions files, checked against `carves.json`).
- **GP slots:** every unordered pair of distinct golds. On the KBs, only questions with 2 to 10 golds count.
- **CH slots:** MuSiQue's chain slots.
- **Deciding:**
  - 2Wiki and HotpotQA: GP, in both RK and SRK;
  - MuSiQue: GP and CH, in RK.
- **Beside:** MetaQA and WebQSP (GP, SRK).
- **Not read:** SQuAD, which has one gold a question and so no slot.

### 9.4 Gates

- **G1a: the rebuild reproduces the record's K at the slot ends** (frozen arm, each dataset, in the frozen `nn` row).
  Both shares below must be at least 0.995, counting tie churn as agreement:
  - (i) **Out-pair precision.** The share of each slot end's rebuilt pairs that are in the record's `knn.npz`. A miss
    at a tie row counts as tie churn.
  - (ii) **Slot agreement.** The share of GP (and CH) slots on which the rebuilt K and the record's K agree. A
    disagreement at a tie row of either end counts as tie churn.

  A dataset that fails G1a is not graded. Its new arms wait for a fix, declared here first.
- **G2a: union consistency.** With the record's K, each domain's union share on s1eval must equal UL1's filed
  `grade.json` share for the F0T form, within 0.01 points. This checks that E1a reads UL1's families: E1a reads them
  one at a time, as membership of the slot keys only.
- **G3a: the vectors.** Every shard of the arm is present, and its manifest's row count equals the record's
  (`zenc.vec_shards`). Damaged rows stop the search.
- **Substrate pin,** as in section 3.

### 9.5 Measures

For each dataset, domain, slot kind and arm, over the carve's slots:
- **K alone:** the share of slots the arm's K links.
- **Union:** the share linked by D83's layout with the arm's K: (S, in SRK) ∪ W2_S ∪ F0T ∪ K_arm.
- **K only:** the share linked by K_arm and by none of S, W2_S or F0T.
- **Against the frozen arm:** the paired difference (new arm minus frozen) for K alone and for the union, with
  `zap1.boot_diff` (2,000 resamples of the questions, seed 0) and ABOVE, LEVEL or BELOW at 95%.
- **MuSiQue only:** the share of questions whose chain slots are all linked, by hop count, with its paired difference.
- **Beside:**
  - the record's K;
  - the frozen arm's G1a counts;
  - the share S and R link without K.

### 9.6 What E1a decides

Three expectations, each called HELD or NOT_HELD per new arm:
- **Y1.** The arm's K-alone GP share is BELOW the frozen arm's on each of 2Wiki, HotpotQA and MuSiQue. This is the
  encoder's price in links.
- **Y2.** On each of those three datasets, in each domain, the union gap (new minus frozen) is closer to zero than the
  K-alone gap. That means S and R carry part of the loss.
- **Y3.** The same as Y2, on MuSiQue's CH slots.

**D97** (open). The default is that every arm that passes G3a goes on to E1b and E2. E1a excludes no arm by itself, and
any exclusion is the user's call. D94's rule from E0 still applies: an arm that E0 keeps out of E1 has its E1a rows
withdrawn, and the report reads it as not read.

### 9.7 Rows

All rows run on the host CPU:
- `enc-e1-nn-frozen-<ds>`, for the five datasets. Each runs now, with 8 threads. It builds the text groups (cached at
  `outputs/enc/e1/groups/`), the frozen neighbour rows and G1a.
- `enc-e1-nn-<enc>-<ds>`. Each runs after `enc-encode-<enc>-<ds>` and the frozen row.
- `enc-e1-grade-<ds>`. Each runs after its four `nn` rows and writes `outputs/enc/e1/grade/<ds>.json`.
- `enc-e1-report`. It runs after the five grades and writes `outputs/enc/e1/report.json` and `report.md`.
- **Then the laptop:** fetch, then a results section here.

### 9.8 Development log (disclosed before any E1 number)

- **The self-test.** `zenc1.py selftest` passes on the laptop. It checks:
  - the text groups;
  - the top 5 against brute force on vectors with exact scores, including exact ties and groups whose own vector is
    outside their top 4;
  - slot membership against the record's rule applied to the whole corpus;
  - G1a, both a pass and a constructed miss;
  - the W2_S read against `zul1.csr_keys`;
  - the expectations.
- **The frozen rows on two datasets.** They ran on the laptop on MuSiQue and MetaQA as a development check of the
  stand-in and of the family reader. These are the only E1a code runs before this page was committed. They printed:
  - **MuSiQue:** 117,533 text groups, equal to the manifest's 117,533 distinct source rows. 2,417 questions give
    5,962 GP slots and 3,987 CH slots over 2,629 slot ends.
  - **MetaQA:** 43,234 text groups over 43,234 positions. 9,785 questions give 37,446 GP slots.
  - **G1a passed exactly on both:** out-pair precision 1.0, slot agreement 1.0, with no tie churn.
  - **G2a reproduced UL1's filed F0T shares exactly:** MuSiQue GP 33.63 and CH 49.71, MetaQA 13.81. These are UL1's
    numbers, not new ones.
- **No new arm has been read.** No new arm's vectors exist yet, so no E1a measure exists. The laptop's outputs were set
  aside; the host rows build their own.

## 10. E1b, part 1: each encoder's base pool B (declared 18 Nov 2026, 20:58 IST, before any E1b number)

Script: `outputs/mp_unified/zenc1b.py` (new; zenc.py and zenc1.py are imported unchanged). Outputs only under
`outputs/enc/e1b/`. Nothing trains.

### 10.1 Why this part first

The user chose the pool check before any E2 fit: "Train E2 only for encoders whose pools keep the golds within about 1
point of frozen", built on MuSiQue, SQuAD and MetaQA on the CPU.

A real pool under another encoder has never been built. D83 kept the record's pools by design (A2). The PKO rule reads
the encoder in four places: the question vectors (the walks and steering), the node vectors (cos1 and kt_cos), the
first-stage lists (B), and the K family. So the full pools need an arm package view and the arm's K over the whole
corpus. That is part 2 (10.6), and it is built only for an arm this part carries.

Part 1 rebuilds the one piece that needs no new machinery: **B, the pool's base**. B is the record's own construction
over the first-stage lists, plus the seeds:
- the base rows `m3b_compile.base_rows` gives for the dataset's own construction (MuSiQue `equal_rrf_budget_200`,
  SQuAD and MetaQA `equal_rrf_budget_50`): equal RRF (constant 60) of the dense and SPLADE top 200, the first 200 or 50;
- the seeds, `m3b_pools.seeds_of`: dense top 5 then SPLADE top 5;
- B = the union of both, as `zpko.rule_batch` forms it.

Each arm's B uses its own dense top-1000 (E0's `outputs/enc/e0/<arm>/<ds>.npz`) and the package's SPLADE top-1000.
Every callee is the record's, unchanged.

**What B is to the pool.** On SQuAD the own cell is rrf50+none, so the pool is B: part 1 is the pool check there. On
MuSiQue and MetaQA the pool is B plus the expansion through the links. The frozen arm's filed expansion share (pool
minus B, from PKO's filed s1eval parts) is read beside, so the size of what the links add is on the record.

### 10.2 Gates (any failure stops the dataset with no measure)

| gate | test |
| --- | --- |
| G1 | the four arms' E0 files hold the same ids, rows and gold totals; the gold cache's ids equal them; every id is in PKO's filed s1eval pools |
| G2 | B rebuilt from the package's own dense list equals the filed B (`bs_nodes`) for every question, exactly |
| G3 | B rebuilt from E0's frozen list equals the filed B for at least 99% of questions, and its gold share is within 0.1 point of the filed B's |

G2 shows the rule is the record's. G3 bounds the float churn of E0's own search, which every arm shares.

### 10.3 Measures (per dataset, s1eval, each arm)

- **gold share of B:** golds held in B over all golds (the ratio over questions), and the share of questions with
  every gold in B;
- the mean size of B;
- **the gap:** the arm minus E0's frozen arm, in points, by `zap1.boot_diff` (2,000 resamples of the questions,
  seed 0), with ABOVE, LEVEL or BELOW;
- beside, for the frozen arm only: the filed B's and the filed pool's gold shares, and X, the expansion share (pool
  minus B, in points).

### 10.4 The call (per arm)

Per dataset:
- **PASS** iff the gap's lower bound is above -1.0 point.
- **NEAR** iff not PASS, X > 0 on that dataset, and the point gap is above -(1.0 + X). The links add X points to the
  frozen pool, so a gap that small could be won back by the expansion, and only the real pools tell.
- **FAIL** otherwise.

Per arm, over MuSiQue, SQuAD and MetaQA:
- **CARRY** iff PASS on all three.
- **NEAR** iff no FAIL and at least one NEAR.
- **HOLD** iff any FAIL.

### 10.5 What follows

- **CARRY:** part 2 (10.6) is declared for that arm, then E2 on its pools.
- **NEAR:** whether part 2 runs for it is the user's call.
- **HOLD:** the arm is not trained (an E2 fit would only measure a loss already seen at the base). Its vectors go on a
  delete list at the filing; the delete is the user's.
- **No arm CARRIES or is NEAR:** E2 does not run. mpnet's 2Wiki encode stays held. The encoder stays the frozen one,
  so D84's MultiHop-RAG encode can use it (the user's call). The encoder gap's next form is the first stage itself.

### 10.6 Part 2 (named only, declared here before its code)

The arm's real pools under PKO's rule on s1eval for MuSiQue, SQuAD and MetaQA. This needs:
- an arm package view (the arm's question and document vectors and dense top-1000 in place of the record's);
- the arm's K over the whole corpus under the record's kNN rule;
- PKO's chain run on it, with every check against the record re-pointed at the view.

The estimate is two to three days of code and about 7 host hours per arm on MuSiQue.

### 10.7 Rows

- `enc-e1b-base-<ds>` for MuSiQue, SQuAD and MetaQA: host CPU, 4 threads, minutes each; all inputs are already on the host.
- `enc-e1b-report`: after the three. It writes `outputs/enc/e1b/report.json` and `report.md`.

### 10.8 Results (filed 18 Nov 2026, 21:06 IST)

The rows ran on the host CPU from 21:04 to 21:05 IST, in 8 to 22 s each. The record is `outputs/enc/e1b/report.md` and
`.json`, with one `<ds>.json` per dataset.

**Gates.**
- G1 passed.
- G2 passed on all three datasets: the rebuilt B equals PKO's filed B on every question (MuSiQue 2,417, SQuAD 11,873,
  MetaQA 9,785).
- G3: E0's frozen list gives the filed B on every question (equal share 1.0), so the arms' gaps carry no float churn.

Shares are % of golds in B. Gaps are points against the frozen arm (95%).

| dataset (frozen B / filed pool / X) | arm | gold share | gap | gate |
| --- | --- | --- | --- | --- |
| MuSiQue (76.06 / 97.77 / +21.71) | qwen3e06 | 72.83 | -3.23 [-3.89, -2.58] BELOW | NEAR |
| | mpnet | 73.08 | -2.98 [-3.62, -2.32] BELOW | NEAR |
| | mpnetqa | 72.24 | -3.83 [-4.45, -3.20] BELOW | NEAR |
| SQuAD (97.98 / 97.98 / 0) | qwen3e06 | 97.09 | -0.88 [-1.08, -0.70] BELOW | FAIL |
| | mpnet | 96.88 | -1.09 [-1.31, -0.88] BELOW | FAIL |
| | mpnetqa | 97.02 | -0.96 [-1.16, -0.76] BELOW | FAIL |
| MetaQA (1.42 / 98.84 / +97.42) | qwen3e06 | 0.86 | -0.56 [-0.66, -0.45] BELOW | PASS |
| | mpnet | 1.08 | -0.34 [-0.46, -0.22] BELOW | PASS |
| | mpnetqa | 0.67 | -0.75 [-0.85, -0.65] BELOW | PASS |

**The calls: qwen3e06 HOLD, mpnet HOLD, mpnetqa HOLD.** Every HOLD comes from SQuAD.

**What it says.**
- **SQuAD.** The pool is B, so no link can win the loss back under the pool rule. Each smaller encoder loses about one
  point of SQuAD's golds there for good. qwen3e06's point gap (-0.88) is inside one point, but its bound misses by 0.08.
- **MuSiQue.** Each arm loses 3.0 to 3.8 points at B. The frozen arm's links add 21.71 points on top of B, so only
  part 2's real pools can say whether the expansion wins those points back.
- **MetaQA.** B holds 1.4% of the golds. The KB links carry the pool (97.42 points), and every arm is within one point.

**What follows (10.5).**
- E2 does not run.
- mpnet's 2Wiki encode stays held.
- The encoder stays the frozen one, so D84's MultiHop-RAG encode can use it.

**D99** (open; the user's call): carry qwen3e06 to part 2 anyway, given the user's "within about 1 point" and its
-0.88 point gap on SQuAD? The default is no: the declared rule holds, and the encoder stays frozen.

## 11. E1b, part 2: each encoder's real pools under PKO's rule (declared 18 Nov 2026, 21:45 IST, before any part 2 pool number)

Script: `outputs/mp_unified/zenc1b2.py` (new). zenc, zenc1, zenc1b, zpko, zpkoh, zu2m_a_types and every module they open
are imported unchanged. Outputs only under `outputs/enc/e1b2/`. Nothing trains.

### 11.1 Why, and what changed since 10.5

Part 1 put all three arms on HOLD, and 10.5 said a HOLD arm goes no further. The user then asked, on 18 November:
"check properly how much can we recover and what are the pools and results after that if it is doable then we will go
with the smallest possible encoder which doesnt lose much". This answers D99 more widely than it asked: part 2 runs for
all three arms, and the choice of encoder moves to 11.5 and 11.6.

Part 1 measured B only. On SQuAD the pool is B, so part 1 already measured SQuAD's pool. On MuSiQue and MetaQA the links add
21.71 and 97.42 points to the frozen pool, so only the real pools say how much of B's loss the expansion wins back.

### 11.2 The view

Each pools run is PKO's own `zpko.pools_pass` on s1eval, with PKO's own rule and opener. Before it opens the carve, the
script swaps the encoder's inputs in the process. The package on disk is never written.

| what the rule reads | the record | an arm's view |
| --- | --- | --- |
| document vectors (cos1, kt_cos, the dense node reader) | the package's | the arm's (`outputs/enc/<arm>/<ds>/docs`) |
| question vectors (the walks, steering) | the package's | the arm's (`outputs/enc/<arm>/<ds>/queries`, every query) |
| the dense top-1000 (seeds, B, steering, anchors) | the package's cache | E0's list for the s1eval rows (`outputs/enc/e0/<arm>/<ds>.npz`) |
| the K family | the package's `graph/knn.npz` and its cached store | this file's K for the arm, built into a store in memory |
| the EM types (U2m (a)) | `outputs/u2m/a/types/<ds>` | refit on the view by U2m's own stage, into `outputs/enc/e1b2/types/<view>/` |

**How.**
- The package loader's `Dataset.embeddings`, `Dataset.cache` and `Dataset.graph` are patched for the one dataset.
- `m3b_pools.load_or_build_store` builds the K store from the view's graph and never saves it. The record's CSR cache is
  not touched.
- `zu2m_a_types.OUT` points at the view's types. `zpko.u2s_record` hands open_pko the view's types sha, so that U2s's
  check reads the view's types.
- Step 4d's union is swapped for the form B1P used (`zb1p.on_b1`): B is the base united with the seeds, and the pools
  are the prepared ones. Step 4d's files hold the record's B, which an arm's B is not.

**What stays the record's:**
- SPLADE;
- the S and R stores and U2's chosen graph;
- U2b's W2_S. Its title tie-break read the dense cosine when it was built; it stays the record's and is disclosed here;
- the score types (U2p);
- the carves, U2s's pick and PKO's rule.

MetaQA's relation vectors are the frozen encoder's. They feed only the types record's NMI field, not the pools.

**Three views beside the arms:**
- **record:** nothing swapped. It rebuilds one filed part per dataset (MuSiQue 0 of 1, MetaQA 0 of 10) for GV.
- **search:** the frozen encoder through every swap an arm takes: E0's frozen list, this file's frozen K, and types refit.
  Every arm is measured against it, so the swaps' own churn cancels.
- **qwen3e06, mpnet, mpnetqa:** the arms.

**K** is the record's kNN rule (zenc1's pieces, 9.2) over every text group. It was built before this section by the first
version of this script (sha256 ab45e320...). That version is committed as is, before this section's version; its K code
is unchanged here. K reads no gold. GK passed on both datasets, the rebuilt frozen edges being the package's:
- MuSiQue: 265,366 edges, weights within 4.2e-7;
- MetaQA: 97,920 edges, weights within 8.9e-7.

### 11.3 Gates (any failure stops the dataset with no measure)

| gate | test |
| --- | --- |
| GK | (passed, above) the frozen K equals the package's `graph/knn.npz`: no pair more or less, weights within 1e-3 |
| GV | the record view's parts equal PKO's filed parts, every array of the npz and the KT rows, exactly |
| GS | the search view's pool gold share is within 0.1 point of the filed pools' (equal-pool share reported) |
| served | in every pools row, every swap served at least once (the cache, the K store, the prepare guard, the union; the vectors for an arm), or the row stops |
| guard | a population row without an E0 list stops the run (the view serves E0's s1eval rows only) |

GT, the search view's refit types against the filed types (array equality, mu's largest difference, cls's equal
share), is reported and is not a gate. GS covers its effect on the pools.

### 11.4 Measures (per dataset, s1eval, each arm, against the search view)

- the pool's gold share (golds in the pool over all golds) and the share of questions with every gold in it;
- **the pool gap** in points, by `zap1.boot_diff` (2,000 resamples of the questions, seed 0), with ABOVE, LEVEL or BELOW;
- B's gold share and gap, read from the same parts;
- **won back** = the pool gap minus B's gap: the points the expansion recovers;
- the mean pool size as a multiple of the search view's;
- beside: the filed pools' and the search view's shares.

### 11.5 The call (per arm)

The call reads MuSiQue's and MetaQA's pool gaps from this part and SQuAD's gap from part 1 (10.8), where the pool is B.

- **KEEPS** iff every lower bound is above -1.0 point.
- **CLOSE** iff not KEEPS, and every point gap is above -1.0.
- **LOSES** otherwise.

**Disclosure.** This rule is written after part 1's SQuAD numbers were read. On those alone mpnet already LOSES (-1.09),
and qwen3e06 (-0.88) and mpnetqa (-0.96) can be CLOSE at best. The -1.0 point is part 1's (10.4) and the user's "within
about 1 point".

### 11.6 What follows

- **The pick:** the smallest arm that KEEPS. If none does, the smallest arm that is CLOSE. The sizes:
  - mpnet and mpnetqa: 109M parameters;
  - qwen3e06: 596M;
  - the frozen encoder: 1.5B.
  Between mpnet and mpnetqa, the one with the higher worst point gap is picked.
- **D100** (the user's call; the default is to go): E2 on the pick. E2 stays named only (section 5). It needs the
  pick's encode of every dataset (2Wiki and HotpotQA included) and its pools on the training carves. It is declared on
  this page, with its cost, before any fit.
- **Every arm LOSES:** the encoder stays the frozen one, and MultiHop-RAG enters the fold on it.

### 11.7 Rows (host CPU, 2 threads a row as PKO's lanes; no GPU)

- `enc-e1b2-types-<view>-<ds>`: search and the three arms, on MuSiQue and MetaQA (8 rows, minutes each).
- `enc-e1b2-dev-mpnet-<ds>`: `--limit 4` on each dataset. Nothing is written but a view record with the timing and
  the served counts. The full rows wait on them.
- `enc-e1b2-pools-record-<ds>`: GV's two parts.
- `enc-e1b2-pools-<view>-<ds>-<i>`: search and the three arms; MuSiQue in 1 part (about 1.5 h each, about 10 GB),
  MetaQA in 10 (about 35 min each, about 7 GB).
- `enc-e1b2-report`: after all of them. It writes `outputs/enc/e1b2/report.json` and `report.md`.

### 11.8 Development log (disclosed before any part 2 pool number)

- K ran on the host from 21:23 to 21:32 IST (8 rows). GK passed.
- The self-test passed: the K edges equal a brute force on a toy corpus with a shared text group, the patches restore,
  and the call rule is as 11.5 says.
- The laptop holds no E0 list, so the view first runs in the host's dev rows. Those rows report timing and served
  counts, not shares.

### 11.9 Results (filed 18 Nov 2026, 22:55 IST)

Records: `outputs/enc/e1b2/report.json` and `report.md` (zenc1b2.py 14b47465, report in 146 s on the host), with each
part's view record under `outputs/enc/e1b2/pko/<view>/pools/parts/` and the refit types under `types/<view>/`.

**Gates.**
- GV passed on both datasets: the record view rebuilt PKO's filed part exactly.
- GS passed on both. The search view's gold share equals the filed pools' (0.0 points). Its pools equal the filed ones
  for 99.79% of MuSiQue's questions and 99.98% of MetaQA's.

**Pools against the search view** (s1eval; points; paired 95% intervals):

| dataset | arm | pool share | pool gap | B gap | won back | rows (x search) |
| --- | --- | ---: | --- | --- | ---: | ---: |
| MuSiQue (search 97.77) | qwen3e06 | 97.19 | -0.58 [-0.86, -0.30] | -3.23 [-3.89, -2.58] | +2.65 | 1.029 |
| | mpnet | 97.22 | -0.55 [-0.85, -0.24] | -2.98 [-3.62, -2.32] | +2.43 | 0.998 |
| | mpnetqa | 96.97 | -0.80 [-1.11, -0.50] | -3.83 [-4.45, -3.20] | +3.03 | 0.995 |
| MetaQA (search 98.84) | qwen3e06 | 98.60 | -0.24 [-0.40, -0.09] | -0.56 [-0.66, -0.45] | +0.32 | 1.000 |
| | mpnet | 97.65 | -1.19 [-1.45, -0.93] | -0.34 [-0.46, -0.22] | -0.85 | 0.988 |
| | mpnetqa | 97.04 | -1.80 [-2.10, -1.49] | -0.75 [-0.85, -0.65] | -1.05 | 0.981 |

**The calls (11.5), with SQuAD's gap from part 1:**
- **qwen3e06: CLOSE.** Every point gap is above -1: MuSiQue -0.58, MetaQA -0.24, SQuAD -0.88. SQuAD's lower bound
  (-1.08) keeps it from KEEPS.
- **mpnet: LOSES.** MetaQA -1.19 and SQuAD -1.09.
- **mpnetqa: LOSES.** MetaQA -1.80.

**The pick (11.6): qwen3e06** (596M parameters, 1024 dims). No arm KEEPS, and it is the only arm that is CLOSE.

**What it shows.**
- On MuSiQue, the links win back most of what a smaller encoder loses. The dense lists (B) lose 3.0 to 3.8 points,
  and the pools lose 0.55 to 0.80.
- On MetaQA, qwen3e06 loses little anywhere. The 109M arms lose more in the pools than in B, so the KB's pool rule
  leans on the encoder there.

**Next: D100, E2 on qwen3e06.** It is the user's call; the default is go. E2 stays named until it is declared on
this page, with its cost, before any fit. It needs qwen3e06's encode of every dataset, 2Wiki and HotpotQA included,
and its pools on the training carves. MultiHop-RAG (D84) enters the fold on the encoder that E2 settles.

## 12. E2: the models on qwen3e06 and on the smallest encoder (declared 18 Nov 2026, 23:08 IST, before any E2 number)

### 12.1 Why, and which arms

The user, after 11.9: "yeah but also even if we lose more than 1 point i want to see how much we can recover even with the
smallest encoder so can we test properly and then decide?"

Section 11 measured the pools. E2 measures the trained models.
- **Arms:** qwen3e06 (11.9's pick, 596M) and mpnet (109M, the smallest).
- **Why mpnet, not mpnetqa:** 11.6's rule between the two 109M arms picks the higher worst point gap in 11.9: mpnet
  -1.19, mpnetqa -1.80. mpnetqa is not carried.
- **Who decides:** the user, after E2's report (D101). Nothing is picked automatically. D100 (E2 on 11.9's pick) is
  widened by the user's words to both arms.

### 12.2 What is rebuilt per arm, and what is kept

Everything the encoder feeds is rebuilt from the arm's vectors, under the record's rules unchanged:
1. **The first-stage lists:** the top 1,000 by the arm's question and document vectors, for every question of D83's five
   fit carves and six s1eval carves (E0's search, extended past s1eval).
2. **K:** the record's kNN rule (zenc1b2's K code) on all six corpora.
3. **The EM types:** refit as in 11.2.
4. **The pools:** PKO's pools pass on the five fit carves and the six s1eval carves. This is 11.2's in-process view,
   extended to all six datasets and the fit carves.
5. **D83's chain on those pools:** look, cache, rmatch and zrc, labelled links, typed, kt, record and gate, with the view
   installed in every step.
6. **In a world per arm (the d83world pattern):** LTM's lists, fold fits and NEXT; LAB; the seed-0 fits. On the GPU, zspk,
   then zg6r, then GNX; on the CPU, zrckpn. Both splits, as in D83. Then the reads on the six s1eval carves.

Kept unchanged: the corpora, questions and carves; S; R's rule links (W2_S and F0T, which read no vector); D83's labels
and layout; and the models' code, epochs, widths, seeds and candidate (p@swa).

The view serves the arm's vectors wherever a step reads a vector. Each step records the vectors it was served. A frozen
vector read inside an arm's run stops that run (11.2's served counts).

### 12.3 The reference

The reference is D83's own seed-0 fits of the same names, read on D83E's six s1eval carves, as filed by D83's report. Both
sides share D83's layout and differ only in the encoder.

### 12.4 Measures (per arm, per s1eval carve, seed 0)

- **The model gap:** R@5 per question (the record's read, p@swa), arm minus reference. The interval is the paired
  bootstrap of zap1.boot_diff (2,000 resamples, seed 0, 95%), with ABOVE / LEVEL / BELOW. It is read for GNX and zrckpn,
  with zg6r beside.
- **Beside:** FC@5 and hit@1, and MuSiQue by hops.
- **Recovery:**
  - the first-stage gap: R@5 of the first-stage list's top 5, arm minus frozen, on the same questions;
  - the pool gap: gold share of the pool, as in 11.4;
  - won back = the model gap minus the first-stage gap.
- **Cost:** parameters, dimensions, document vectors on disk, the encode's measured GPU hours, and the query encode time
  per question on the host's CPU (cold, measured).

### 12.5 What the report says

- **Per arm, 11.5's words applied to the model reads** (GNX and zrckpn on the six carves): KEEPS, CLOSE or LOSES.
- **Beside it, the recovery and the cost.**
- **No pick is made.** The user chooses the encoder that carries (D101).
- **After the choice:** E3 (three seeds on the chosen encoder), then MultiHop-RAG (D84) encoded with it.

### 12.6 Gates (a failure stops the arm's step; none of its numbers is read)

- **GE0:** the arm's lists on the s1eval carves equal E0's (`outputs/enc/e0/<arm>/<d>.npz`).
- **GK:** as in 11.3, on all six: the frozen rebuild equals the package's K.
- **GS:** a search view (the frozen encoder through every swap) rebuilds one fit carve's pools. They must equal the
  pools of record within 11.3's bar, so the view itself changes nothing.
- **Served counts:** each step's record lists every vector read and its source.
- **D83's own gates (section 8)** run unchanged in each chain.

### 12.7 Rows, order, disk and cost (estimates; the host's measured numbers replace them)

1. **GPU encodes, after D91's last GPU row:** first `enc-encode-mpnet-2wiki` (about 1 h; mpnet's other datasets are
   encoded), then qwen3e06 on WebQSP, HotpotQA and 2Wiki (about 8-9 h). New vectors: about 9 GiB and 27 GiB.
2. **Code:** `outputs/mp_unified/zenc2.py` and an `e2world` runner, both new files; nothing of D83's is edited. Each is
   self-tested on the laptop and committed before its step runs.
3. **mpnet first, since its vectors land first, then qwen3e06:**
   - per arm, the CPU chains take about 1 to 1.5 days on 3 lanes;
   - per arm, the GPU fits take about 8 h.
4. **Disk:** about 25 GiB per arm beyond the vectors (D83's chain and world measured 16 GiB beside its linked pools).
   - The host had 111 GiB free at declaration.
   - The arms run one after the other.
   - Each arm's look and cache are pruned after its reads (16-17 Nov ruling), and the 30 GB floor is kept.
5. **ETA for the report:** about 24-25 November 2026, after D91's GPU rows.

### 12.8 Development log (disclosed 23:22 IST, before any E2 number)

**GE0's bar.** zenc2.py's first bar (77971e0) asked for the whole 1,000-id lists to equal E0's on 99.9% of the s1eval
rows. On mpnet it failed on SQuAD (97.46% of rows) and MuSiQue (94.62%). On both, every top-5 set was equal and the
position-wise scores agreed within 0.0005, which is float16's step. The lists differ only by the order of near-ties deep
in the list: E0 searched the s1eval rows alone, and E2 searches every query, so the GEMM blocks differ.

The bar is replaced, before any list is used, by this rule. On E0's s1eval rows:
- every top-5 set is equal on 99.95% of the rows;
- the position-wise scores agree within 0.002;
- the lists are equal up to near ties on 99.95% of the rows. An id in both lists must score within 0.002 on both sides.
  An id in one list only must score within 0.002 of the lower list end.

The share of equal whole lists is reported but not gated. `zenc2.py gate` applies the rule to a saved list file and keeps
the first verdict as `ge0_first`.

**GE0's second form (23:31 IST).** On mpnet's MetaQA lists, the rule above failed only on its top-5 share: 99.73%,
against the 99.95% bar.
- No row differed beyond a near tie.
- The scores agreed within 0.00024.
- MetaQA's KB holds many entities with near-identical text, so the 5th place often ties.

The top-5 share is now reported, not gated. What is gated is the share of rows whose top-5 sets differ by more than a
near tie at the 5th place: an id in one top 5 only, scoring above the lower 5th score + 0.002. That share must be at
most 0.05%.

The other two parts are unchanged:
- the position-wise scores within 0.002;
- the lists equal up to near ties on 99.95% of the rows.

The rows already gated under the first form (SQuAD and MuSiQue, both arms) pass both forms.

### 12.9 Amendment: in-domain first, on the small and hard datasets (declared 18 Nov 2026, 23:43 IST, before any E2 number)

**Why.** A survey of the code (18 Nov, about 23:35) found three things.

First, the model stack assumes 1536-wide vectors in about ten places:
- look's projection;
- lean_mlp's SEMB and projection;
- rmatch and zu2n's Q_DIM;
- zg6r's P_f and P_b;
- DenseNodes;
- lean_mlp3.fit_basis.

zg6r's encoder element count is checked against a pinned number.

Second, every chain step runs in its own process.

Third, an exact K over HotpotQA (5.2M passages), 2Wiki (6.0M) and WebQSP (2.6M) costs about a day of CPU per dataset
per arm. That runs against the user's "check only the small and hard datasets ... dont waste space nor time". The user,
on the plan: "we will check in domain first as comparision then any generalization stuff".

**A correction, disclosed.** A note to the user (18 Nov, about 23:40) said the models read the old twin0/gnn0 scores.
They do not. Every fit opens the cache without them, and a read loads them only as reference rankings printed beside the
candidates (lean_gpu.read_carve). The record has one model per family: zrckpn (MLP) and GNX (GNN). GNX warm-starts from
zg6r, which warm-starts from zspk; these are stages of the one GNN, not separate models. Look still writes those
reference scores, and nothing trains on them.

**This replaces 12.2-12.7 where they differ.**

**Worlds.** Small worlds with one recipe:
- the frozen world: the reference, retrained the same way;
- mpnet (the smallest);
- qwen3e06, only when mpnet's call below is LOSES or the user asks.

**Carves.**
- Training: SQuAD fit and MuSiQue s1fit (D83F's carves of the two).
- Reads, in-domain: SQuAD and MuSiQue s1eval.
- Generalization reads (MetaQA s1eval, the big datasets, MultiHop-RAG) come after, each declared before its numbers.

**The frozen world.** It links D83F's and D83E's chain outputs for the two datasets (sha-equal; the pools of record). It
retrains LTM, LAB and the fits on the two training carves. Its reads are the reference; D83's own five-carve fits are not.

**Width.** The arm's float16 rows are copied into 1536-wide rows with zeros after them:
- every inner product, norm and top-k list stays the arm's own;
- the record's code runs with unchanged widths and parameter counts;
- a padded column is zero in every row, so no weight on it can change an output.

The arm's real width is in the cost (12.4). A real-width build is for the encoder the user chooses.

**Rebuilt per arm (the padded vectors):**
1. The first-stage lists over every query: zenc2 first, GE0 PASS (12.8).
2. K: SQuAD's from zenc2; MuSiQue's from E1b part 2.
3. The kNN store, built from that K in the arm's own folder. The record's CSR cache is never opened: it loads with no
   check against the graph.
4. The EM types, refit.
5. The basis: the record's rule (fit_basis on the 60,000 sampled nodes of the record's basis graph, seed 0), on the
   arm's vectors of those rows, encoded for this.
6. PKO's pools on the four carves.
7. D83's chain on them.
8. LTM, LAB, then zspk -> zg6r -> GNX and zrckpn, at seed 0 on one split (a screen fit, section 5).

**Measures (per s1eval carve).**
- The model gap: R@5 per question, the arm's world minus the frozen world. zap1.boot_diff gives the paired interval and
  ABOVE / LEVEL / BELOW, for GNX and zrckpn, with zg6r beside.
- Beside it: the first-stage gap (the dense top 5), the pool gap, won back (the model gap minus the first-stage gap) and
  the cost.

**The call (per arm, information only).** 11.5's words (KEEPS / CLOSE / LOSES) on the two in-domain reads, for GNX and
zrckpn. The user decides (D101).

**Gates.**
- GE0 (12.8), GK and D83's own gates are unchanged.
- GW: the frozen world's chain inputs are D83's (sha-equal).
- Every vector the arm's carves read comes from the arm's files. A frozen dense row read for SQuAD or MuSiQue inside an
  arm's world stops it.

**Cost and ETA.**
- Disk: the padded SQuAD and MuSiQue vectors are about 1 GiB; a world is about 5 to 8 GiB.
- The four big encodes stay held (feed 793a17ec).
- Code first, about a day. The worlds run after D91's GPU rows. The report is about 21-22 November 2026.
