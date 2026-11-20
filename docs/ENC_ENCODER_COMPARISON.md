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

### 12.10 Amendment: the arm's pools on the host (declared 19 Nov 2026, 00:35 IST, before any E2 number)

**What a first run showed.** Two audited pools passes ran on the host, 20 SQuAD fit questions each, nothing written.
- **The frozen world** ran (0.98 s per question at 2 threads). The record files it read fall in three groups:
  - files the arm replaces: the dense vectors and lists, the CSR kNN store, the EM types, and step 4d's pools;
  - files kept as the record's (below);
  - verdicts, carves and the host mirror's verify records.
- **The mpnet world** stopped at step 4e's check ("base and seeds are not step 4d's"). Step 4d's pools were built from
  the frozen lists.

**How a step runs.** The record's PKO host runs use zpkoh's mirror route: the package is served from the host mirror,
in memory, with S6.host_mode's pinned verify records. E2 uses the same route. In an arm's world, zpko runs as a module
through `outputs/mp_unified/zenc2p.py` (`types`, `pools`, `merge`, each with `--host`). The hook patches modules as they
load, and a main script is not one.

**What the hook swaps.** This makes 12.9's list exact. For SQuAD and MuSiQue in an arm's world:
- **The package loader:** the dense vectors, the dense lists and K, as in 12.9.
- **M3B's kNN store:** built in memory from the arm's K. The record's CSR file is never opened.
- **The EM types:** zu2m_a_types writes to and reads from the arm's folder, `outputs/enc/e2/types/<arm>/`. U2s's record
  carries the arm's types sha, as zenc1b2 swapped it in 11.2.
- **Step 4e's union:** B1P's form. B is the base united with the seeds, and I is the prepared pools, as E1b part 2 built
  its pools (11.2).
- **The bases:** zpko.copy_bases takes them from `outputs/enc/e2/bases/<arm>/`.

The arm's spec forbids reading the record's EM types, step 4d's pools of the two datasets, and the two bases. zenc2p
checks, before it writes, that every swap its step needs is in force.

**Kept as the record's.** These files are encoder-free, or they are recipe constants:
- the score types (U2p) and the SPLADE lists, as zenc1b2 kept them in E1b part 2;
- U2b's chosen graph W2_S. It reads no dense vector: the dense context choice is the `_D` variants, and W2_S differs from
  W2_DS on MuSiQue;
- M3B's NER store;
- U2s's and U2e's stats on the s1sel carves (the budgets and walk-mass counts), which are the recipe's constants.

**The frozen side's form, disclosed.** The frozen world keeps the pools of record (12.9), and the arm uses B1P's union
form. 11.9's GS measured the gap between the two forms with the frozen vectors:
- gold share: 0.0 points;
- equal pools: 99.79% of MuSiQue's s1eval questions.

The report states this beside the model gap.

**The bases.** fit_basis runs on the record's draw (60,000 rows, seed 0) at two BLAS threads. Its input is the arm's
float16 rows, padded to 1536:
- **HotpotQA:** rows from the arm's full encode (host GPU, float16 model).
- **2Wiki:** the drawn rows' texts, encoded for this on the host CPU (float32 model, 6 threads). 2Wiki's full encode
  stays held (12.9).

The two devices differ only by rounding. The record's draw-only reader gives the full reader's basis bit for bit
(zenc2p self-test).

**Padding, made exact.** A padded inner product equals the unpadded one in exact arithmetic. In float32 it agrees up to
rounding, not bit for bit, because the GEMM's blocking depends on the width. 12.9's "stays the arm's own" means this.

**Order.**
- **W1:** the arm's bases, EM types and pools on the four carves (feed rows `e2p-*`). A 20-question check runs before
  the parts.
- **W2:** D83's chain in the arm's world, then the training.

The training code fixes four training datasets per split (zwq.SPLITS, and the carve-count checks in the column builders).
12.9's two-carve training therefore needs a patched split. It is declared in 12.11 before any fit.

**Gate added.**
- **GA:** every E2 run's audit summary shows 0 blocked reads and 0 blocked writes.
- A failed run's numbers are not read.

### 12.11 Amendment: the two-carve training (declared 19 Nov 2026, 01:13 IST, before any E2 fit)

**Why.** 12.9 trains on two carves (SQuAD fit, MuSiQue s1fit). The record's training code fixes four training datasets
per split (zwq.SPLITS). Tables and checks built at import read that table: ltmcols.ALL_CARVES (11 carves), zwqp's and
zltg's asserts. D83's training and read steps run only in a tree named d83world with D83WORLD set (zd83t, zd83m and
zd83r check both). Neither D83's code nor the training code is edited; both are pinned.

**The split, patched by the hook.** When an E2 spec names a split (`outputs/mp_unified/e2hook/sitecustomize.py`):
- while the process runs, zwq's SPLITS["WQ-musique"] is (squad, musique) and ZERO_SHOT["WQ-musique"] is (metaqa,
  hotpotqa, 2wiki, webqsp);
- while any module executes (at any depth), both tables hold the record's entries. Every table and check built at import
  is therefore the record's, unchanged;
- lean_gpu.basis_of gives 2wiki for a training set with neither 2Wiki nor HotpotQA. The record's rule gives 2wiki for
  both record splits, so the basis graph is the record's (in an arm's world, 12.10's arm basis of it);
- the key stays "WQ-musique", so every file name is the record's form (fit names without "-hp");
- the hook checks the table when zwq loads: both keys exist, training plus zero-shot is zwq.EVAL_ORDER, and every
  training dataset has a fit carve. A bad split is refused. A spec without a split behaves as before (self-test).

Two consequences, stated before any fit:
- ltmcols' MIN_LISTS (200) is unchanged. SQuAD fit has no A lists, so LTM fits on MuSiQue s1fit's lists alone.
- ZERO_SHOT is only written into the fits' metadata (zltm). No training step reads a zero-shot carve.

**The training worlds.** Each E2 world gets one: `<E2 world>/e2t/d83world`, made by sworld's make with d83world's
targets and the E2 world as its root, in train mode (outputs/pko -> outputs/abl/D83F). Its code is pinned to the E2
world's copies, and zenc2w pins those to the record's. Every step runs through `outputs/mp_unified/zenc2t.py run`: the
hook, the audit, the write guard and GA, as in 12.10. `--where e2` runs a step in the E2 world itself (an arm's LAB and
D83 gates).

**What each world holds.**
- **Frozen:** its outputs/abl/D83F and D83E are junctions to the record's (the pools of record, 12.9). Its LAB tables
  and D83 gates are therefore the record's, already done. zenc2w's world command refuses those junctions, so the frozen
  world is refreshed with `zenc2t.py world --enc frozen` (which also refreshes its training world).
- **mpnet:** its own D83F and D83E (12.10's W2 chains), its own gates (`zd83.py gate --host`), and its own LAB on the
  four carves.

**Steps per world** (seed 0; the record's commands and sizes, run in the training world):
1. LTM lists on four carves: SQuAD fit, MuSiQue s1fit, SQuAD s1eval, MuSiQue s1eval.
2. The LTM fit (`ltm fit --split WQ-musique --host`).
3. The columns on the same four carves.
4. The fits: zspk -> zg6r -> GNX on the GPU, zrckpn on the CPU.
5. The forks, then six reads (arm D83E): zg6r, GNX and zrckpn on SQuAD s1eval and MuSiQue s1eval. zg6r is a stage of
   the one GNN, read beside GNX (12.9), not a third model.

**Gate GW, the fence.** Every D83F or D83E entry of a dataset other than SQuAD and MuSiQue is a forbidden read, the
record's and the E2 world's (to depth 3). In an arm's world the record's whole D83F and D83E are forbidden. A step that
touches one stops, and none of its numbers is read.

**Development log (disclosed before any E2 fit).**
- W1's 20-question check (feed row e2p-a1) ran on the host: 20 SQuAD fit questions, 1.28 s per question, 0 blocked
  reads, 8 swaps in force (the SQuAD kNN store among them), nothing written. zenc2p then stopped it with "the swap of
  the knn store is not in force". The check was wrong, not the run. zu2m and zu2c rebind m3b_pools.load_or_build_store
  to their own wrappers, so the attribute is no longer the hook's function itself. zenc2p's check now follows a
  wrapper's closure to the hook's function, and checks every loaded copy of the module. Its self-test passes a wrapper
  that reaches the hook and refuses one that does not. The check runs again (e2p-a1b) before the parts.
- D83's chain reads K through the package loader, so its kNN family comes from the swap (read in the code).

**The report.** Its code is written before the first read lands: per world and carve, R@5 per question; the arm minus
the frozen world with zap1.boot_diff (12.9); the first-stage gap, the pool gap, won back and the cost beside. The call
uses 11.5's words. The user decides (D101).

**Order.** The frozen world's rows go in now: they need nothing from W1 or W2. Its GPU fits yield to other users, as
every GPU row does. mpnet's rows wait on its W2 chains. Estimates: the frozen reads about 20 November 2026; mpnet's
about 21 November; the report as 12.9 says.

**Development log, 19 Nov 2026 02:13 IST (before any E2 number).**
- **The report's code** is zenc2r.py (3cd6585): `collect` on the host copies each world's reads whose sha256 is their
  marker's, and the s1eval pool parts behind them; `report` on the laptop reads them as above. GA leaves a failed run
  out only when its named successor passed (zenc2r.SUPERSEDED: the two runs below).
- **The frozen world's zrckpn fit** (t-fz-fit-zrckpn) trained for 23 minutes and then stopped with rc 1. After training,
  the fit's screen.json is stamped with the sha256 of every dataset's chain record under the tree (rmatch_records), and
  the fence refused the first of WebQSP's. Nothing of it was read. Every fit stamps the same walks (the built_records of
  rmatch, zlink, relcols, hubwalk and zfeat). With a split, the hook now narrows them to the split's datasets (SQuAD and
  MuSiQue): no other dataset's record is opened, and an E2 fit's screen.json lists only those two datasets' chain
  records. zenc2t's self-test checks it: a WebQSP record beside SQuAD's and MuSiQue's is left out with the split, and
  listed without it. The fit reruns from scratch as t-fz-fit-zrckpn-b; the frozen GNN fits were held until the change
  was in place.
- **mpnet's first D83F chain** (w2b-chain-f-squad) stopped after 9 s at D83's look step. PKO builds nothing before its
  section-5 checks pass (<base>/check: WebQSP, HotpotQA, SQuAD fit and SQuAD s1eval), and the arm's PKO base has none.
  Those checks certify PKO's code (it rebuilds the earlier pools); the arm's world runs that code pinned to the
  record's, and re-running them on the arm's vectors would compare against the frozen vectors' pools by design. The arm's
  world therefore carries the record's four PASS verdicts (zenc2k.py: sha256-checked copies, and carried.json naming
  their sources). The chain reruns as w2c-chain-f-squad.
- **The rerun (w2c-chain-f-squad)** passed the check gate and stopped after 8 s at D83's store: ABL's slot types
  (outputs/abl/g/<dataset>/<slot>/types.json; D83F fills its structural slot with R, D83E with S) are not in the arm's
  world. They are EM types fit on the slot graph's node vectors (ABL section 4, U2m (a)'s types stage), so the arm fits
  its own with ABL's command, unchanged: `zabl_g.py all musique`, then `all squad`, in the arm's world (the hook swaps
  the vectors and U2m's types as in 12.10). The chain reruns as w2d-chain-f-squad after both. The frozen world uses
  the record's D83F and D83E, built with the record's slot types.
- **ABL's slot types, first run (w2d-g-musique)**: zabl_g stopped after 77 s while writing its record. It records
  each path relative to the tree it runs in, and the arm's U2m types folder (the hook's swap, 12.10) lies in the record's
  tree. zabl_g now runs through zenc2g.py, which imports it unchanged and records such a path relative to the world
  ("../ws/..."). The fits rerun as w2d-g2-musique and w2d-g2-squad. The report's tag families and reruns now live in
  docs/ENC_E2_RUNS.json (data, not pinned code), so naming a rerun no longer touches the worlds.
- **The chain after the slot types (w2d-chain-f-squad)** stopped after 147 s in D83's KT pass, at ABL's check 1.
  ABL's on_arm installs its own step 4e union. Its form is 12.10's: B is the base united with the seeds, I is the
  prepared pools. Check 1 then compares each question's base and seeds with step 4d's filed carve. That carve holds the
  frozen vectors' pools, which the arm's spec forbids, so the guard refused the read and nothing was built. For an arm
  dataset, the hook now keeps ABL's form and runs check 1 against the arm's own filed carve: the world's PKO pools from
  12.10's W1, with the sha256 checked against their manifest. A record dataset keeps ABL's function, and the frozen
  world swaps nothing. zenc2t's self-test checks all three cases. The chain reruns as w2e-chain-f-squad.
- **The frozen world's zg6r fit (t-fz-fit-zg6r)** ran out of GPU memory three times (rc 1 after 409 s; peak 12.85 GB
  under a 13.2 GiB cap). The training rows kept the GPU shares of the record's four-dataset split (zg6r 0.47, GNX 0.52).
  E2's split holds MuSiQue s1fit, as D91's -hp split does, and D91's -hp rows ask for cuda_alloc 0.9 and a 0.85 share.
  E2's zg6r and GNX fits in both worlds now ask the same. Nothing else changes: same code, data and seed. zspk passed at
  its own share. The fit reruns from scratch as t-fz-fit-zg6r-b.
- **mpnet's D83F MuSiQue chain (w2b-chain-f-musique)** ended rc 1 after 6,891 s. The world had no MuSiQue label sets
  yet, so the six look lanes that start together each wrote the same labels/musique/1.npy through os.replace, and two
  of them (shards 3 and 4) lost that race at 02:59 (WinError 5). The other 24 shards ran to the end, so the race had
  left retry_cmd's 4,000-character window long before the chain ended. zd83 build skips filed steps, and
  write_label_sets compares a label file already on disk instead of writing it. The chain reruns as w2c-chain-f-musique
  and redoes shards 3 and 4 only. The two D83E chains will start on empty label folders, so they now run under
  e2hook/e2race.py instead of retry_cmd. It is retry_cmd's rule (rerun on WinError 5 or 32, up to three runs, any other
  failure ends at once) with the whole output scanned. It sits in the junctioned e2hook folder and touches no pinned
  file. Its self-test covers the three cases. Nothing else changes.
- **qwen3e06's world starts (19 Nov, 20:21 IST).** 12.9 runs it "only when mpnet's call below is LOSES or the user
  asks". An interim collect and report (20:13, GA PASS on both worlds) gives mpnet's zrckpn call as LOSES: SQuAD
  -2.77 [-3.18, -2.35], MuSiQue -3.82 [-4.72, -2.95]. mpnet's GNX reads are still queued behind the GPU, so its GNX
  call is INCOMPLETE. The world starts now, on one deciding model's LOSES, because a pick between the arms needs
  qwen3e06's cells whatever GNX gives, and the host's CPU is idle. The user may stop it or say otherwise (D101).
  The rows are mpnet's last generation with --enc qwen3e06, in the same order and at the same sizes: pads, world,
  bases, types, pools, ABL's slot types, the four chains, then the training rows, forks and reads, and a collect.
  Three things differ:
  - No whole-graph qwen3e06 encode of HotpotQA or 2Wiki exists (12.9: the small and hard datasets only), so both
    bases CPU-encode the 60,000 drawn rows, as mpnet's 2Wiki basis did (2,088 s at six torch threads). qwen3e06 has
    about 5.5 times mpnet's parameters, so each takes about three hours. The two run side by side.
  - All four chains run under e2race, since the world's label folders start empty for D83F too.
  - The slot-type rows wait for the world's pools, as mpnet's ran.
  After the four chains, the world's filed look chunks are pruned (prune_d83_looks.py, the 16-17 Nov rule). The
  world's build tags start q1- and its training tags t-qw- (docs/ENC_E2_RUNS.json). No qwen3e06 number is read before
  its report.
- **The GPU share of the E2 GNN fits (19 Nov, 21:55 IST).** mpnet's zg6r fit waited at the head of the GPU queue
  because another user's idle context held about 2.3 GB of the card, leaving 0.842 free against the row's 0.85.
  The four unsent zg6r and GNX fit rows (mpnet's and qwen3e06's) now ask the feeder for 0.82. cuda_alloc's cap stays at
  0.9. The frozen world's zg6r and GNX fits peaked at 15.4 GB allocated (20.55 GB reserved). The code, data and seed
  are unchanged.
- **mpnet's calls (20 Nov 2026, 00:51 IST).** mpnet's zg6r and GNX fits, forks and reads (t-mp-*-b) ran, then
  the collect (e2r-collect-mp-b) and the report. GA passes in all three worlds. Both deciding models LOSE:
  - GNX: SQuAD -2.62 [-3.02, -2.16], MuSiQue -5.14 [-6.06, -4.23].
  - zrckpn: SQuAD -2.77 [-3.18, -2.35], MuSiQue -3.82 [-4.72, -2.95], the same as the interim report.
  - zg6r, read beside: SQuAD -2.59 [-3.01, -2.15], MuSiQue -4.25 [-5.18, -3.35].
  - The first stage loses 14.18 (SQuAD) and 11.89 (MuSiQue) points. The models win back 11.41 to 11.59 and 6.75 to
    8.07 of them. The pools hold 1.09 and 0.55 points fewer golds.
  mpnet's arm is complete. qwen3e06's cells are INCOMPLETE until its world's reads. The user decides (D101).
- **qwen3e06's first D83F chain (q1-chain-f-squad, 20 Nov 01:48)** stopped after 5 s at PKO's check gate. Its world
  had no section-5 verdicts: mpnet's world got them from `zenc2k.py carry --enc mpnet`, which ran outside the feed, so
  the cloned rows left it out. `zenc2k.py carry --enc qwen3e06` ran at 01:53 (four sha256-checked copies and
  carried.json; the code unchanged). zenc2w refuses a tag that holds a run, so the feeder's retry of the row runs
  under q1-chain-f-squad-b (docs/ENC_E2_RUNS.json: superseded). Nothing else differs.

- **qwen3e06's calls; E2 complete (20 Nov 2026, 08:11 IST).** qwen3e06's fits, forks and reads ran, then
  e2r-collect-qw and the report (outputs/enc/e2/report.md). GA passes in all three worlds. Both deciding models LOSE:
  - GNX: SQuAD -1.49 [-1.88, -1.09], MuSiQue -3.58 [-4.47, -2.71].
  - zrckpn: SQuAD -1.67 [-2.03, -1.29], MuSiQue -3.91 [-4.76, -3.10].
  - zg6r, read beside: SQuAD -1.51 [-1.90, -1.11], MuSiQue -2.19 [-3.10, -1.35].
  - The first stage loses 6.33 (SQuAD) and 4.46 (MuSiQue) points. The models win back 4.66 to 4.84 on SQuAD but only
    0.55 to 0.88 on MuSiQue (2.27 for zg6r). The pools hold 0.88 and 0.58 points fewer golds.
  - Against mpnet (-2.62 / -5.14 GNX, -2.77 / -3.82 zrckpn), qwen3e06 loses less on SQuAD and about as much on MuSiQue.
  Every E2 cell is read: mpnet LOSES and qwen3e06 LOSES on both deciding models. No pick is made (12.5). The user
  decides (D101).

## 13. E2M: mpnet's vectors mapped into the frozen space, at the first stage (declared 20 Nov 2026, 01:00 IST, before any map is fit)

Code: outputs/mp_unified/zenc2m.py (run, report, selftest). Outputs: outputs/enc/e2m/. The user asked for it on 20 Nov
("sure check it out"). It reads nothing of qwen3e06 and changes nothing of E2.

### 13.1 Why

mpnet's E2 arm loses with both deciding models (12.11's log, 20 Nov 00:51). Most of the loss starts at the first
stage: mpnet's dense top 5 is 14.18 (SQuAD) and 11.89 (MuSiQue) points behind the frozen encoder's, and the models win
back 6.75 to 11.59 of those points. E2M asks whether that first-stage loss is geometry, which a map learned from
mpnet's space into the frozen encoder's recovers, or information, which no map recovers. If a map recovers enough,
the smallest encoder could stand in for the frozen one under the models as they are trained, and a new dataset would
need only mpnet's encode.

### 13.2 The pairs

- SQuAD and MuSiQue pooled, from the vectors on the host (the record's frozen rows and the mpnet rows E0 encoded; no new encode).
- Questions: every train-split question (SQuAD 130,319; MuSiQue 19,938), mpnet's vector to the frozen one.
- Documents: every document that is no dev or test question's gold. Every s1eval gold is among those left out (the
  run checks). Dev questions' other documents stay in, so the eval golds are the only documents the maps never see.
- 10% of each side (seed 0) is held out and picks each arm's setting by mean cosine. No s1eval question and no eval
  gold enters a fit or a choice.

### 13.3 The arms

Two maps per arm: f_q for questions and f_d for documents. The outputs are L2-normalised.
- **ridge:** least squares with a bias (the bias unpenalised); lambda from 0.01, 0.1, 1, 10, 100 or 1,000 by
  held-out cosine.
- **mlp:** ridge's map plus a residual branch (768 to 2,048, GELU, to 1,536) whose last layer starts at zero, so epoch
  0 is ridge. AdamW at 1e-3 (weight decay 1e-4 on the branch only), batch 2,048, loss 1 minus cosine, 60 epochs. The
  held-out best epoch is kept (epoch 0 included). Seed 0.

### 13.4 The reads

- On the s1eval carves E0 and E2 read (SQuAD 11,873 questions, MuSiQue 2,417), with E0's measure: zenc.search
  (exact float32, top 1,000), R@5, R@20 and R@1000, and RRF@5 with the record's SPLADE list.
- **both:** f_q(mpnet question) against f_d(mpnet document) over the whole corpus. This is the small encoder alone,
  and it decides.
- **qonly:** f_q(mpnet question) against the frozen documents. Beside: only the question side is small.
- **G1:** the run's own raw mpnet read equals E0's mpnet hits on at least 99.5% of the questions at every k, per
  carve. Otherwise the report stops.

### 13.5 The call

Per arm, on 'both': R@5 minus E0's frozen R@5 in points per carve, paired 95% bootstrap over questions (2,000, seed
0). Won back is the same against E0's mpnet read.
- **ADVANCE:** both carves' point gaps clear one deciding model's bars. A bar is -1 point divided by that model's E2
  ratio of model gap to first-stage gap on the carve: zrckpn SQuAD -5.12 and MuSiQue -3.11; GNX SQuAD -5.41 and
  MuSiQue -2.31. It assumes the models win back the same share of the first-stage gap as they did on raw mpnet. It
  is a gate for spending more, not a claim about the models.
- **PARTIAL:** not ADVANCE, but won back's interval starts above 0 on both carves. The map recovers a real part.
- **STOP:** otherwise. mpnet's first-stage loss is taken as information in its vectors.
- The stage's call is ADVANCE if either arm's is, else PARTIAL if either arm's is, else STOP. Both arms are reported.
- Beside, with no call: R@20, R@1000 (the pools' reach), RRF@5, the qonly reads, and the held-out cosines.

### 13.6 What follows

- ADVANCE: a stage 2 is declared on this page and put to the user before it runs. The frozen world's trained models
  would read on the mapped vectors, in a world built like E2's, with no training.
- PARTIAL or STOP: no stage 2. The result goes beside E2's for D101.

### 13.7 Rows and cost

- e2m-selftest (CPU), e2m-run (one GPU row: 8 threads, about 20 GB, 0.29 of the card; estimated under 30 minutes),
  then e2m-report (CPU).
- The maps take about 60 MB and the reads about 20 MB, under outputs/enc/e2m/. The mapped corpus (up to 0.72 GB) is
  a temporary file the run removes.

### 13.8 Report (20 Nov 2026, 01:08 IST): STOP

- e2m-run ran in 257 s on the host's GPU. G1 passed: the run's raw mpnet read equals E0's on every question at every
  k on both carves.
- The pairs: 150,257 questions and 133,934 documents (SQuAD's 1,204 and MuSiQue's 2,629 dev or test golds left out).
  Held-out cosine to the frozen vectors: questions 0.749, documents 0.725 (ridge, lambda 0.1).
- **ridge, both** (decides): R@5 SQuAD 66.98, gap to frozen -20.72 [-21.54, -19.91]; MuSiQue 22.75, -22.97
  [-24.12, -21.77]. Against raw mpnet it loses more: won back -6.54 [-7.18, -5.95] and -11.08 [-11.99, -10.18].
  qonly is worse still (-23.76 and -30.68 to frozen).
- **mlp** never beat its start. The first epoch lowered the held-out cosine (0.749 to 0.745 for questions, 0.725 to
  0.722 for documents), and 60 epochs climbed back to just under epoch 0. The best epoch is 0, so the mlp arm is ridge
  and its reads equal ridge's. The recipe's step size (1e-3 on top of the ridge solution) is the likely cause. A
  better-tuned branch could add a little cosine, but closing a 20-point gap to the -5.12 bar is not within reach of
  that, so it is not re-run.
- **Calls:** ridge STOP, mlp STOP; the stage is STOP. A map fit to the frozen vectors recovers none of mpnet's
  first-stage loss: the mapped space retrieves worse than mpnet's own. Under 13.5, mpnet's first-stage loss is taken as
  information in its vectors. There is no stage 2. The result sits beside E2's for D101.
- What it does not test: a map trained on retrieval itself (questions against their gold documents). That is in-domain
  supervised training on top of mpnet, the job the E2 models already do (they win back 6.75 to 11.59 points).

## 14. E2S: can mpnet's vectors be improved without touching mpnet? (declared 20 Nov 2026, 06:40 IST, before any E2S number)

Code: outputs/mp_unified/zenc2s.py (selftest, fuse, adapt, winenc, win, report). Outputs: outputs/enc/e2s/. The user,
20 Nov: "first we should test all this directly if the vectors can even be improved then see the next steps". Each arm
asks one question at the first stage, and nothing is declared after it. No arm changes mpnet's weights.

### 14.1 Why

- E0 (mpnet minus frozen, points): dense R@5 -14.18 on SQuAD and -11.89 on MuSiQue; dense R@1000 -1.03 and -7.29. On
  SQuAD, mpnet's top 1,000 holds the gold almost as often as the frozen encoder's: the loss is the order at the top.
- Ruled out already: truncation (mpnet's encode cut 167 of SQuAD's 20,233 documents and 214 of MuSiQue's 117,534);
  smoothing the vectors over the links (SEM1, below its base for mpnet); a map into the frozen space (E2M, 13.8).

### 14.2 The arms

- **win (several vectors per document).** mpnet's model card: "The sequence length was limited to 128 tokens" in
  training, and input "longer than 384 word pieces is truncated". Each document is cut into windows of W content tokens
  at stride W/2, at the character offsets of mpnet's fast tokenizer; a document of W tokens or fewer is one window (its
  whole text). W is 64 or 126. On the host's count, 61.5% of SQuAD's documents and 24.0% of MuSiQue's are longer than
  126 tokens (94.1% and 63.6% longer than 64): 39,407 and 159,298 windows at 126, and 86,113 and 301,495 at 64. The
  windows are encoded by zenc.encode_texts (mpnet, GPU, fp16, as E1). A document scores
  (1 - beta) cos(q, d) + beta max over its windows cos(q, w), beta from 0, 0.25, 0.5, 0.75 or 1.
- **fuse (mpnet with mpnetqa).** E1's whole-document vectors of both 110M models: (1 - w) cos_mpnet + w cos_mpnetqa,
  w from 0, 0.25, 0.5, 0.75 or 1.
- **adapt (a learned map on the question vector; the documents untouched).** f(x) = unit(x + g(x)) on mpnet's question
  vector. g is linear (768 to 768) or a residual MLP (768 to 2,048, GELU, to 768). g's last layer starts at zero, so
  epoch 0 is raw mpnet. The document index stays mpnet's.
  - Pairs: SQuAD's and MuSiQue's train-split questions, one pair per gold. Left out: the s1sel questions, and every
    question that shares a gold with an s1sel question. No dev or test question enters (no dev gold of either dataset
    is a train question's gold).
  - Loss: softmax cross-entropy at temperature 0.05 over the batch's gold documents and the question's 16 highest-ranked
    non-gold documents in raw mpnet's top 50 (mined once); the question's other golds are masked. AdamW, weight decay
    0.01, batch 512, 10 epochs, seed 0. Each batch holds one dataset's pairs.
  - The setting: g's form, the learning rate (1e-4, 3e-4 or 1e-3) and the epoch (0 to 10), by s1sel (14.3).
  - Beside, with the chosen form and learning rate: maps trained on SQuAD alone (read on MuSiQue) and on MuSiQue alone
    (read on SQuAD), each epoch by its own dataset's s1sel; and the same recipe on the frozen vectors (1,536 wide, hard
    negatives from the record's dense list), read on SQuAD and MuSiQue.
  - **Unseen read:** the chosen map on HotpotQA's s1eval (7,405 questions), a dataset it never saw, over mpnet's whole
    5.2M-document HotpotQA index.

### 14.3 Selection

One setting per arm, by the mean of SQuAD's and MuSiQue's s1sel dense R@5 (1,498 and 1,534 train-side questions, the
carves SEM1 chose on; no s1eval question). Ties go to the setting nearest raw mpnet: the smaller beta, w or epoch, then
the smaller W, the linear form and the smaller learning rate. If raw mpnet's own setting wins, the arm reads as raw
mpnet.

### 14.4 Reads

- On E0's s1eval carves (SQuAD 11,873, MuSiQue 2,417): exact float32 search over the whole corpus, top 1,000; dense
  R@5, R@20 and R@1000, and RRF@5 with the record's SPLADE list (E0's measure).
- **G1:** the run's raw mpnet read (raw mpnetqa as well for fuse; raw mpnet on HotpotQA for adapt) equals E0's hits on
  at least 99.5% of the questions at every k. Otherwise the report stops.
- **G2 (win):** each one-window document's window vector against E1's vector of that document: cosine 0.999 or more on
  at least 99.5% of them, per dataset. Otherwise the report stops.

### 14.5 The call

Per arm, on s1eval dense R@5: won back is the read minus E0's raw mpnet read, in points, paired 95% bootstrap over
questions (2,000, seed 0).
- **IMPROVES:** won back's interval starts above 0 on both carves.
- **MIXED:** above 0 on one carve, and not below 0 on the other.
- **NO:** otherwise.
- adapt must also hold on HotpotQA: if its won back there is BELOW (the interval under 0), the call is **IN-DOMAIN
  ONLY**, whatever the two carves say.
- Beside, with no call: the gap to E0's frozen read and the share of it won back; that gap against E2M's bars (13.5)
  as a pointer only; R@20, R@1000 and RRF@5; the cross reads, the frozen map's reads and the selection curves. Three
  arms are read, so the chance that one passes by luck is above 5%.

### 14.6 What follows

Nothing is declared. The calls go to the user, who sets the next step.

### 14.7 Rows and cost

- e2s-selftest (CPU); e2s-fuse (CPU, 8 threads, minutes); e2s-adapt (CPU, 8 threads, about 1 to 2 h with HotpotQA's
  search); e2s-winenc (GPU 0.29, behind qwen3e06's fits in the GPU queue, minutes); e2s-win (CPU); e2s-report.
- The window vectors take about 0.9 GB under outputs/enc/e2s/win/ (rebuildable). The maps take a few MB.

## 15. E2N: node memory on HotpotQA (declared 20 Nov 2026, 06:40 IST, before any E2N number)

Code: outputs/mp_unified/zenc2n.py (selftest, run, report). Outputs: outputs/enc/e2n/. The user, 20 Nov: "if we are able
to improve pools eventually each nodes vectors would be better over the course of multiple questions". E2N asks whether
a document's vector improves from the questions that reached it.

### 15.1 Why HotpotQA alone

A document can learn only from questions that reach it. Dev questions with a gold that some train-split question also
has: SQuAD 0 of 11,873; MuSiQue 0 of 2,417; HotpotQA 6,026 of 7,405 (81.4%; 41.6% have every gold seen, and 59.6% of
HotpotQA's dev golds are train golds). On SQuAD and MuSiQue the answer is 0 by construction, so they are not read.

### 15.2 The forms

A document d with a feedback set Q_d becomes unit(d + lambda unit(mean of Q_d's question vectors)). Documents with no
feedback are unchanged. lambda is 0.25, 0.5, 1 or 2.
- **gold:** Q_d = HotpotQA's train-split questions with d among their golds: what a perfect ranker of the pools would
  feed back.
- **self:** Q_d = the train-split questions whose raw dense top 1 is d. No labels.
- s1sel's questions (train-side) are left out of every feedback set. No validation or test question enters one.
- Encoders: mpnet (decides) and frozen (beside: whether memory closes the gap or lifts both). frozen's top 1 is the
  record's dense list.

### 15.3 Selection and reads

- lambda per form and encoder, by HotpotQA's s1sel dense R@5 (1,508 questions; 18.2% hold a fit-carve gold, as 18.3%
  of s1eval's do). Ties go to the smaller lambda.
- Reads on HotpotQA's s1eval (7,405): dense R@5, R@20 and R@1000, and RRF@5 with the record's SPLADE list. Exact: the
  raw top 2,000 over the whole index, with every changed document rescored. R@5 and R@20 are exact. R@1000 is exact
  where at least 1,000 unchanged documents sit in the raw top 2,000; the share is reported.
- **G1:** the run's raw reads equal E0's HotpotQA hits (mpnet's and frozen's) on at least 99.5% of the questions at
  every k. Otherwise the report stops.

### 15.4 The call

Per form, on mpnet's s1eval dense R@5 against E0's raw mpnet read, paired 95% bootstrap (2,000, seed 0): **HELPS**
(the interval above 0), **LEVEL** or **HURTS**. Beside, with no call: the share of mpnet's 23.14-point HotpotQA gap to
frozen won back; the split by whether a question has a gold seen in training (any gain should sit with that 81.4%);
frozen's own won back; R@20, R@1000 and RRF@5.

### 15.5 What follows

Nothing is declared. The calls go to the user.

### 15.6 Rows and cost

e2n-selftest, e2n-run (CPU, 8 threads, about 2 to 3 h: the self form searches the 5.2M documents for HotpotQA's 90,447
train questions, and frozen's reads search its 1,536-wide index), e2n-report. The output is a few MB.

### 14.8 Report (20 Nov 2026, 08:00 IST): win IMPROVES, fuse IMPROVES, adapt NO

From `zenc2s.py report` (outputs/enc/e2s/report.md; the arms' records fuse.json, adapt.json, win.json and
win/winenc.json). G1 passed in every run (the raw reads equal E0's on every question at every k, HotpotQA included). G2
passed: one-window documents' window vectors against E1's, minimum cosine 0.9999 (SQuAD) and 0.9996 / 0.9981 (MuSiQue
at W 64 / 126). The windows: SQuAD 86,113 (W 64) and 39,407 (W 126); MuSiQue 301,495 and 159,298; encoded in 287 s.

Won back is the read minus raw mpnet's (E0), s1eval dense R@5, points, 95% paired bootstrap:

| arm (setting, by s1sel) | SQuAD won back | MuSiQue won back | share of E0's gap won (SQuAD / MuSiQue) | call |
|---|---|---|---|---|
| win (W 64, beta 0.5) | +5.05 [+4.58, +5.53] | +0.86 [+0.38, +1.36] | 36% / 7% | IMPROVES |
| fuse (w 0.5) | +2.96 [+2.56, +3.39] | +1.52 [+0.96, +2.08] | 21% / 13% | IMPROVES |
| adapt (linear, lr 1e-4, epoch 4) | -0.88 [-1.38, -0.38] | +4.46 [+3.66, +5.29] | -6% / 38% | NO |

- **adapt on HotpotQA, never trained on:** +3.38 [+2.88, +3.89] over raw mpnet (46.01 against 42.63), so the NO is not
  IN-DOMAIN ONLY. Its pairs were 118,903 (SQuAD) and 36,564 (MuSiQue). Every one of the six adapter settings beat
  raw mpnet on the s1sel mean (0.540 to between 0.554 and 0.576).
- **Beside: cross maps.** SQuAD's map read on MuSiQue +2.43 [+1.88, +3.00]; MuSiQue's map read on SQuAD -3.42 [-3.94,
  -2.89]. The same recipe on the frozen vectors: SQuAD -0.87 [-1.24, -0.50], MuSiQue +2.17 [+1.46, +2.91]. The map
  helps multi-hop questions and costs SQuAD a little, on both encoders.
- **Beside: the windows.** On s1sel, W 64 beats W 126 on SQuAD at every beta, on MuSiQue up to beta 0.5. On SQuAD the window score alone (beta 1) is
  already above the document vector, and the mix is best; on MuSiQue windows help only a little and alone they hurt.
- **Beside: what is left.** Every arm's gap to frozen stays far outside E2M's bars (win -9.13 / -11.03; fuse -11.22 /
  -10.37; adapt -15.05 / -7.43; the bars need about -5 / -3).
- **What the calls say:** yes, mpnet's vectors can be improved without touching mpnet. Each method wins back a part of
  the gap, and each helps a different place: windows help SQuAD's single-paragraph questions, the question map helps
  multi-hop questions (MuSiQue, and HotpotQA unseen), and fusing two small encoders helps both a little. The three were
  read one at a time; how they combine was not tested. Three arms were read, so one lucky pass is more likely than 5%;
  two passed, each on both carves.
- Nothing is declared after E2S (14.6). The user sets the next step.

### 15.7 Report (20 Nov 2026, 08:12 IST): gold HELPS, self LEVEL

From `zenc2n.py run` (4,704 s on the host's CPU) and `report` (outputs/enc/e2n/report.md, run.json). G1 passed for both
encoders. Feedback came from 88,939 train-split questions (s1sel's left out). 81.1% of the 7,405 s1eval questions have
a gold among those questions' golds ("seen"). Every R@1000 read was exact (the share is 1.0).

| encoder | form | lambda | changed docs | R@5 | won back [95%] | seen | not seen |
|---|---|---|---|---|---|---|---|
| mpnet | gold | 1.0 | 104,398 | 55.27 | +12.65 [+11.92, +13.37] | +16.46 | -3.67 |
| mpnet | self | 0.25 | 73,622 | 42.57 | -0.05 [-0.38, +0.26] | +0.20 | -1.14 |
| frozen | gold | 0.5 | 104,398 | 74.08 | +8.31 [+7.79, +8.81] | +10.84 | -2.53 |
| frozen | self | 0.25 | 70,992 | 66.63 | +0.86 [+0.61, +1.12] | +1.23 | -0.75 |

- **Calls (15.4, mpnet):** gold **HELPS**, self **LEVEL**.
- **gold** wins back 55% of mpnet's 23.15-point HotpotQA gap to frozen (the gap left: -10.50 [-11.30, -9.68]). It lifts
  the frozen encoder too (+8.31), so memory raises both encoders rather than closing the gap. The gain sits with the
  seen questions. Questions whose golds no training question reached lose 3.67 (mpnet) and 2.53 (frozen) points: moved
  documents crowd them.
- **self** does nothing for mpnet, whose own top 1 is a gold for 54.2% of the feedback questions. For frozen, whose top 1
  is a gold 81.6% of the time, it helps a little (+0.86). On s1sel every larger lambda was worse for self, and lambda
  1 was best for mpnet's gold feedback.
- **What the calls say:** a document's vector does get better from the questions that reached it, if the feedback is
  right. Feedback as noisy as raw mpnet's top 1 is not enough. Between the two sit the deciding models' own top picks
  (hit@1 about 87 to 91 on B1's HotpotQA settings in D83's reads, with the frozen encoder); they were not read here.
- Nothing is declared after E2N (15.5). The user sets the next step.

## 16. E2B: the best vectors per encoder, before the encoder is chosen (declared 20 Nov 2026, 11:50 IST, before any E2B number)

Code: outputs/mp_unified/zenc2b.py (selftest, winenc, maps, grid, hotpot, report). Outputs: outputs/enc/e2b/. The user, 20
Nov, after 14.8 and 15.7: "do vectors find the best ways only then we can say if we can take a particular encoder". E2B
combines the parts per encoder, picks each encoder's best combination on s1sel, and reads it once on s1eval. D101 waits
for it. No encoder's weights change.

### 16.1 Why

- E2S read each part alone, on mpnet only. win and fuse IMPROVE; adapt helps MuSiQue and the unseen HotpotQA but costs
  SQuAD 0.88. Each helps a different place. Their combination was not read, and none was read on qwen3e06.
- E2's cells (12.4) show where the loss enters. On SQuAD, mpnet's pools hold almost every gold (96.88 against frozen's
  97.98), yet GNX loses 2.62 points. The loss comes through the vectors the models read, not through the pools. So a
  part counts for the models only if it leaves one vector per question and one per document.

### 16.2 The arms

- **mpnet** (109M), with mpnetqa (109M) as its fusion partner: 218M if fusion is chosen.
- **qwen3e06** (596M).
- **frozen** (1.5B), beside: the same parts on the record's encoder. It shows whether the parts close the gap or lift
  both.

### 16.3 The parts (per encoder of the arm)

- **q-map:** f(x) = unit(x + gamma g(x)) on the question vector, with gamma 0, 0.5 or 1 (1 is the map as trained).
  - mpnet's and frozen's maps are E2S's (adapt.npz, adapt-frozen.npz, checked by sha).
  - qwen3e06's and mpnetqa's are trained by `maps` with E2S's recipe: 14.2's pairs and loss, hard negatives from the
    encoder's own top 50, linear, learning rate 1e-4, the epoch (0 to 10) by the s1sel mean, as E2S's frozen map. The
    run stops if a pair's document is an s1eval gold.
- **fuse (the mpnet arm only):** (1 - w) S_mpnet + w S_mpnetqa, w from 0, 0.25, 0.5, 0.75 or 1. As one vector,
  [sqrt(1 - w) v_mpnet ; sqrt(w) v_mpnetqa] (1,536 wide) gives the same score.
- **prf (the question side, no labels):** after a first pass, q' = unit(q + alpha unit(mean of the first pass's top k
  documents' vectors)), per encoder, with k 3 or 10 and alpha 0.25, 0.5 or 1. The second pass scores with q' in every
  term.
- **win (several vectors per document):** S = (1 - beta) cos(q, d) + beta max over d's windows cos(q, w), with W 64 or
  126 and beta 0 to 1 in steps of 0.25.
  - The windows are E2S's spans (mpnet's tokenizer), the same texts for every encoder. `winenc` encodes them as
    documents (no prompt). mpnet's are E2S's.
- **Two recipes per arm:**
  - **V** uses q-map, fuse and prf only. It keeps one vector per question and per document, so the models can read it.
  - **F** adds win. F's windows can change the first-stage lists, but they are not a vector.

### 16.4 Selection (per arm, on s1sel: the mean of SQuAD's and MuSiQue's dense R@5, exact over the whole corpus)

- **Step A:** every (gamma, W, beta, w) without prf. mpnet has 3 x 9 x 5 = 135 settings; qwen3e06 and frozen have 27.
  V's setting is the best with beta 0; F's is the best of all.
- **Step B:** for V and for F, prf off or one of the six (k, alpha), on step A's setting.
- Ties go to the setting nearest raw: the smaller gamma, beta and w, then W; prf off, then the smaller alpha and k.

### 16.5 Reads (E0's s1eval carves: SQuAD 11,873 questions, MuSiQue 2,417)

- V and F per arm, exact over the whole corpus, top 1,000: dense R@5, R@20 and R@1000, and RRF@5 with the record's SPLADE
  list.
- **Leave one out:** each recipe with one of its chosen parts set back to raw.
- **G1:** each arm's raw setting equals E0's read of that encoder (mpnetqa's too) on at least 99.5% of the questions at
  every k.
- **G2 (winenc):** each one-window document's window vector against its document vector, cosine 0.99 or more on at
  least 99.5% of them. This is E0's G0 bar; 14.4's 0.999 was for mpnet's own encode, and frozen's document vectors come
  from the record's encode. The window index must equal E2S's.
- A failed gate stops the report.

### 16.6 HotpotQA, beside (no call)

- mpnet and frozen only. qwen3e06 and mpnetqa have no HotpotQA vectors, and windows would need a new encode of 5.2M
  documents.
- The reads use V's gamma and prf (mpnet's w is set to 0), with and without E2N's gold memory (15.2, at its lambda:
  mpnet 1.0, frozen 0.5).
- Exact as in 15.3: the top 2,000 over the whole index, with every changed document rescored. With memory, prf's first
  pass takes the memory vectors of changed documents.
- **G4:** the run's raw-plus-memory read equals E2N's on at least 99.5% of the questions at every k.

### 16.7 The call (per small arm; s1eval dense R@5; paired 95% bootstrap over questions, 2,000, seed 0)

- **The recipe:** won back against E0's raw read of the arm, in 14.5's words (IMPROVES, MIXED or NO), for V and for F.
- **The gate, on V:** the gap to E0's frozen read against the arm's bars. A bar is -1 point divided by the model's E2
  ratio of model gap to first-stage gap on the carve (13.5), taken from E2's report for that arm.
  - mpnet's bars are 13.5's: zrckpn -5.12 (SQuAD) and -3.11 (MuSiQue), GNX -5.41 and -2.31.
  - qwen3e06's come from 12.11's cells: about zrckpn -3.79 and -1.14, GNX -4.25 and -1.25 (the report computes them).
  - **ADVANCE:** both carves clear one deciding model's bars.
  - **PARTIAL:** not ADVANCE, and V's won back starts above 0 on both carves.
  - **STOP:** otherwise.
- **Beside, with no call:**
  - F against the bars, as a pointer;
  - each small arm's gap to frozen's own V and F (whether the gap closes or both rise);
  - the leave-one-out drops;
  - R@20, R@1000 and RRF@5;
  - the cost: parameters, vector width, windows per document, and prf's second search.
- One read per arm and recipe on s1eval. Every setting is chosen on s1sel.

### 16.8 What follows

- **ADVANCE:** stage 2 is declared on this page and put to the user before it runs. The arm's E2 world would be rebuilt
  on V (12.2's view serving V's vectors), and GNX and zrckpn refit and read as in 12.4.
- **PARTIAL or STOP:** nothing is trained. The result goes beside E2's for D101.
- Either way, the user decides D101.

### 16.9 Rows and cost (estimates; the host's measured numbers replace them)

- e2b-selftest (CPU).
- e2b-winenc-mpnetqa, then -qwen3e06, then -frozen (GPU). Each encodes 586,313 windows over both W, in about 5 min,
  30 min and 1 h.
- e2b-maps (CPU, about 15 min).
- e2b-grid-mpnet, -qwen3e06 and -frozen (CPU, 8 threads, about 30 to 60 min each).
- e2b-hotpot (CPU, about 2 h), then e2b-report.
- **Disk:** the windows take about 3.9 GB under outputs/enc/e2b/win/. They can be rebuilt, and are kept until D101,
  since stage 2 would need them. The maps and reads take under 100 MB. The host had 59 GB free at declaration.

## 17. E0N: three retrieval-trained encoders at mpnet's size, at the first stage (declared 20 Nov 2026, 12:35 IST, before any E0N number)

Code: `outputs/mp_unified/zenc3.py` (models, encode, first, report, selftest). Records: `outputs/enc/e0n/`. Vectors go
under `outputs/enc/<arm>/<ds>/`, as E0's do, and reads beside E0's under `outputs/enc/e0/<arm>/`.

### 17.1 Why

- The user, 20 Nov: "are there any better models around this size?", then "yes download those three and run the
  screen we will try to improve from there if the scores are workable".
- E2B's mpnet grid (section 16) chose V = q-map at gamma 1 with mpnetqa fused at w 0.5. Its s1eval dense R@5 is 75.43
  on SQuAD and 39.31 on MuSiQue, against frozen's 87.69 and 45.72.
  - On SQuAD, 91% of the gold gap at 5 is order: the golds are inside mpnet's top 1,000 (gap at 1,000: 76 golds,
    against 857 at 5).
  - On MuSiQue the gap at 1,000 (374 golds) is larger than the gap at 5 (327): the golds are never reached.
  - Both point at what the model was trained for (all-mpnet-base-v2: sentence similarity, trained at 128 tokens),
    not at what is done with its vectors.
- Public retrieval scores put several retrieval-trained models of mpnet's size well above it. They are not taken as
  evidence. Qwen3-Embedding-0.6B reports 61.83 on MTEB(eng, v2) retrieval and still loses 6.33 points to frozen on
  SQuAD (E0). This stage reads the three on our carves.

### 17.2 The arms

| arm | model | revision (pinned) | params | dim | length |
|---|---|---|---|---|---|
| `gtemodern` | Alibaba-NLP/gte-modernbert-base | e7f32e3c00f9 | 149M | 768 | 8,192 |
| `granite` | ibm-granite/granite-embedding-english-r2 | 47ea694b257b | 149M | 768 | 8,192 |
| `granitesm` | ibm-granite/granite-embedding-small-english-r2 | 2ab6fa8ea2d6 | 47M | 384 | 8,192 |

- **The text contract is section 2's.** Document text verbatim with no prompt; questions verbatim. None of the three
  publishes a query prompt (gte-modernbert-base's sentence-transformers config lists none; the Granite repos have no
  prompt config). The models step stops if a loaded model has a non-empty prompt or a default prompt.
- **Length:** 8,192 tokens for each, the limit the model cards state, set explicitly. Each config's own default is
  recorded beside. Cut rows are counted and reported.
- **Pins:** the revisions above are the repos' heads on 20 Nov. They are recorded with every file's sha256 in
  `outputs/enc/e0n/models.json`. `outputs/enc/models.json` is not touched.
- Licences: Apache-2.0 for all three. Parameter counts are the cards'; the models step counts them.
- **Encoding as E0's (section 2):** host GPU, fp16, unit L2, float16 shards of 40,000 rows. Every query row and every
  document row of SQuAD (142,192 and 20,233) and MuSiQue (24,814 and 117,534). Only these two datasets (the user, 18
  Nov: "check only the small and hard datasets").

### 17.3 Gates

- **G0c** (section 3) on every shard.
- **GP (precision):** before an arm's encode, 256 document rows and 256 query rows per dataset (rng seed 0) are encoded
  on the GPU in float32 and in float16. Every row's cosine between the two must be at least 0.99. ModernBERT (the base
  of all three) was trained in bf16, so fp16 is checked, not assumed. A failure stops the arm.
- **GR (the reads' references):** E0's frozen, mpnet and qwen3e06 reads and E2B's mpnet-V read hold the same question
  ids in the same order as each new read. Otherwise the report stops.

### 17.4 Reads

`zenc.py first`, unchanged (section 4): E0's s1eval carves (SQuAD 11,873 questions, MuSiQue 2,417), exact search over
the whole corpus. Dense R@5, R@20 and R@1000, and RRF@5 with the record's SPLADE list.

### 17.5 The call (per arm; s1eval dense R@5; paired 95% bootstrap over questions, 2,000, seed 0)

- **NO:** the interval against E0's raw mpnet read does not start above 0 on both carves.
- **WORKABLE:** above raw mpnet on both carves.
- **STRONG:** WORKABLE, and also above E2B's mpnet V read on both carves (the best mpnet recipe a model can read).
- **Beside, with no call:**
  - the gap to frozen, set against mpnet's bars (13.5) and qwen3e06's (16.7) as pointers only, since the models' win-back
    ratio on a new encoder is not measured;
  - the gap to qwen3e06's raw read;
  - R@20, R@1000 and RRF@5;
  - MuSiQue by hop count;
  - the order share of the gold gap to frozen (1 minus the gap at 1,000 over the gap at 5);
  - the cost: parameters, vector width, encode seconds and cut rows.

### 17.6 What follows

- For each WORKABLE or STRONG arm, best first by the mean of its two R@5 gaps to frozen, the next step is E2B's recipe
  search on that arm ("we will try to improve from there"), declared on this page before it runs.
- A NO arm gets nothing further. Its vectors (about 0.47 GB for a 768-wide arm, 0.23 GB for granitesm) are offered to
  the user for deletion.
- Nothing is trained in this stage. The user decides D101.

### 17.7 Rows and cost (estimates; the host's measured numbers replace them)

- e0n-selftest (CPU), then e0n-models (CPU; downloads of about 0.6, 0.6 and 0.2 GB through huggingface_hub).
- e0n-enc-gtemodern, -granite and -granitesm (GPU, 0.29 of the card; GP, then both datasets' queries and documents;
  some minutes each). The GPU queue keeps the file's order, so they run after E2B's window encodes.
- e0n-first-<arm>-<ds> (CPU, 8 threads; six rows), then e0n-report.
- **Disk:** about 1.2 GB of vectors and 1.4 GB of downloads. The host had about 59 GB free at declaration.

### 17.8 Report (20 Nov 2026, 13:08 IST): all three STRONG

- **Gates:** GP passed for all three: the lowest float32-to-float16 cosine was 0.99999 or higher. G0c passed on every
  shard, and no row was cut at 8,192 tokens. GR passed.
- Encode time on the host GPU, all four sets: gtemodern 319 s, granite 320 s, granitesm 67 s.

| arm | params / dim | SQuAD R@5 | MuSiQue R@5 | vs raw mpnet | vs mpnet V | vs qwen3e06 raw | vs frozen | call |
|---|---|---|---|---|---|---|---|---|
| gtemodern | 149M / 768 | 83.09 | 44.27 | +9.57 / +10.44 | +7.66 / +4.96 | +1.72 / +3.01 | -4.61 [-5.19, -4.02] / -1.45 [-2.38, -0.57] | STRONG |
| granite | 149M / 768 | 77.38 | 44.63 | +3.86 / +10.80 | +1.95 / +5.32 | -3.99 / +3.37 | -10.32 [-10.97, -9.65] / -1.09 [-2.12, -0.13] | STRONG |
| granitesm | 47M / 384 | 76.12 | 43.89 | +2.60 / +10.06 | +0.69 / +4.59 | -5.25 / +2.63 | -11.57 [-12.26, -10.90] / -1.82 [-2.81, -0.87] | STRONG |

- Points, SQuAD / MuSiQue. Every interval against raw mpnet and against mpnet V starts above 0 (granitesm against V on
  SQuAD: [+0.02, +1.36]). References: frozen 87.69 / 45.72, raw mpnet 73.52 / 33.83, qwen3e06 81.37 / 41.26, mpnet V
  75.43 / 39.31.
- **gtemodern beats qwen3e06 on both carves** at a quarter of its parameters (149M against 596M). Its raw gap to frozen
  (-4.61 / -1.45) clears mpnet's bars for both deciding models (zrckpn -5.12 / -3.11, GNX -5.41 / -2.31), as a pointer
  only. It does not clear qwen3e06's tighter MuSiQue bars (-1.14 and -1.25).
- **On MuSiQue all three come close to frozen:** -1.09 to -1.82 points. By hop count, gtemodern is above frozen on
  4-hop questions (25.93 against 23.58), and granite is level on 3-hop (41.40 against 41.45).
- **The SQuAD gap is order, as for mpnet:** 90 to 93% of the gold gap at 5 is inside the top 1,000 (gtemodern: 547 golds
  at 5, 57 at 1,000). On MuSiQue the gap at 1,000 is larger than at 5 (gtemodern 308 against 63; granite 97 against 52):
  the deep reach still trails frozen's (R@1000 82.69, 85.49 and 84.27, against 87.25).
- Beside: RRF@5 gtemodern 88.39 / 47.66. That is above frozen on MuSiQue (47.34), and 2.14 points under it on SQuAD
  (90.53).
- **For comparison, from E2B (section 16, its grid read, before E2B's report):** qwen3e06's windows recipe F reads 87.08
  / 43.76, and its one-vector V 82.01 / 42.50. Windows took qwen3e06 within 0.61 points of frozen on SQuAD.
- **What follows (17.6):** E2B's recipe search on the new arms, gtemodern first, is declared next (section 18). Nothing
  was trained. The user decides D101.

## 18. E3B: E2B's recipe search on E0N's encoders (declared 20 Nov 2026, 13:12 IST, before any E3B number)

Code: `outputs/mp_unified/zenc3b.py`. It runs E2B's code (zenc2b.py, unchanged) with its output folder, arms, window and
map encoders and stage name set for this stage, and with fusion for every two-encoder arm. Records: `outputs/enc/e3b/`.

### 18.1 Why

- The user, 20 Nov: "we will try to improve from there if the scores are workable". E0N (17.8) called all three
  encoders STRONG.
- E2B's grid for qwen3e06 showed that windows can close most of a SQuAD gap: its F read 87.08, against frozen's 87.69.
- granite and granitesm come within 1.1 and 1.8 points of frozen on MuSiQue, and gtemodern within 4.6 on SQuAD. Fusing
  gtemodern with one of them may combine the two strengths.

### 18.2 The arms

| arm | encoders (first, second) | params | vector width |
|---|---|---|---|
| `gtemodern` | gtemodern | 149M | 768 |
| `gtemodern+granite` | gtemodern, granite | 298M | 1,536 |
| `gtemodern+granitesm` | gtemodern, granitesm | 197M | 1,152 |
| `granitesm` | granitesm | 48M | 384 |

- granite alone is not an arm: it trails gtemodern by 5.7 points on SQuAD and leads it by 0.4 on MuSiQue. Fused,
  w = 1 is granite alone, so the fused arm's grid holds it.

### 18.3 The parts and recipes: 16.3's, with three changes

- **fuse** applies to both two-encoder arms: (1 - w) S_first + w S_second, w from 0 to 1 in steps of 0.25. As one vector
  it is [sqrt(1 - w) v_first ; sqrt(w) v_second].
- **q-map:** each new encoder gets its own map from `maps`, with 16.3's recipe (E2S's pairs and loss, hard negatives
  from its own top 50, linear, learning rate 1e-4, the epoch by the s1sel mean). The run stops if a pair's document is an
  s1eval gold.
- **win:** each new encoder encodes E2S's windows (`winenc`, G2 as 16.5).
- prf, the recipes V and F, selection (16.4), the reads, leave one out, and G1 and G2 (16.5) are E2B's. Step A has 135
  settings for a two-encoder arm and 27 for one. HotpotQA is not read: the new encoders have no HotpotQA vectors.

### 18.4 The recipe call (per arm and recipe)

14.5's words, on won back against E0N's raw read of the arm's first encoder: **IMPROVES**, **MIXED** or **NO**.

### 18.5 The gate (per arm, on V; the gap to E0's frozen read in points)

- **ADVANCE:** both carves clear qwen3e06's bars for one deciding model (zrckpn -3.79 / -1.14 or GNX -4.25 / -1.25). They
  are the bars of the closest encoder whose models were measured, and the tighter set.
- **NEAR:** not ADVANCE, but both carves clear mpnet's bars for one deciding model (zrckpn -5.12 / -3.11 or GNX -5.41 /
  -2.31).
- **SHORT:** otherwise.
- The bars are pointers. No new encoder's models have been measured, so the share of the first-stage gap the models
  win back is assumed from qwen3e06 and mpnet.

### 18.6 Beside, with no call

- F against both sets of bars.
- V and F against E2B's qwen3e06 V and F, and against frozen's own V and F (E2B's frozen grid).
- The leave-one-out drops; R@20, R@1000 and RRF@5; the cost (parameters, vector width, windows).

### 18.7 What follows

- The arm with the best gate, by the mean of its two V gaps to frozen, is named. If an arm with fewer parameters has the
  same gate and is within 0.5 points of that mean, it is named too ("the smallest possible encoder which doesnt lose
  much", the user, 18 Nov).
- **ADVANCE or NEAR:** that arm's E2 stage is declared on this page and put to the user before it runs. It would rebuild
  the arm's world on V (12.2's view serving V's vectors) and refit and read GNX and zrckpn as in 12.4.
- **SHORT:** nothing is trained.
- The user decides D101.

### 18.8 Rows and cost (estimates; the host's measured numbers replace them)

- e3b-selftest (CPU). e3b-winenc-gtemodern, -granite and -granitesm (GPU, 0.29 of the card; 586,313 windows each,
  minutes each). They queue after E2B's frozen windows.
- e3b-maps (CPU, about 10 minutes for three maps), then four grids (CPU, 8 threads, 10 to 20 minutes each), then
  e3b-report. The report waits for E2B's frozen grid, whose V and F it reads beside.
- **Disk:** windows about 0.9 GB for each 768-wide encoder and 0.45 GB for granitesm (2.3 GB), kept until D101. Maps and
  reads take under 100 MB.

### 18.9 Report (20 Nov 2026, 14:34 IST): both fused arms ADVANCE

- **Gates:** G1 passed on every arm's raw read, on both carves. G2 passed for all three encoders' windows: the lowest
  cosine was 0.99999 or higher, and every one-window document cleared 0.99.
- **The maps:** gtemodern's best s1sel epoch was 1, granite's 6, granitesm's 0 (its map is the identity).
- **Chosen settings:**
  - Both fused arms: V = gamma 0.5 and w 0.5 (equal weights), no prf; F adds windows (W 64, beta 0.75).
  - gtemodern: V = gamma 0.5; F adds W 64, beta 0.75.
  - granitesm: V is raw (no s1sel setting beat its plain vector); F = W 64, beta 0.5.
  - prf was chosen for no arm.

| arm | params / width | V R@5 | V gap to frozen | V call | gate | F R@5 | F gap to frozen | F call |
|---|---|---|---|---|---|---|---|---|
| gtemodern | 149M / 768 | 82.93 / 45.18 | -4.77 [-5.35, -4.21] / -0.54 [-1.49, +0.35] | MIXED | NEAR | 88.15 / 45.95 | +0.45 [-0.08, +0.99] / +0.23 [-0.69, +1.13] | IMPROVES |
| gtemodern+granite | 298M / 1,536 | 83.53 / 49.00 | -4.17 [-4.72, -3.63] / +3.28 [+2.38, +4.18] | MIXED | ADVANCE | 88.71 / 49.09 | +1.01 [+0.49, +1.54] / +3.37 [+2.43, +4.29] | IMPROVES |
| gtemodern+granitesm | 197M / 1,152 | 83.68 / 48.15 | -4.02 [-4.57, -3.49] / +2.43 [+1.53, +3.29] | IMPROVES | ADVANCE | 88.93 / 48.62 | +1.24 [+0.70, +1.77] / +2.91 [+1.96, +3.83] | IMPROVES |
| granitesm | 48M / 384 | 76.12 / 43.89 | -11.57 [-12.26, -10.90] / -1.82 [-2.81, -0.87] | NO | SHORT | 82.59 / 44.62 | -5.10 [-5.69, -4.49] / -1.10 [-2.10, -0.13] | IMPROVES |

- s1eval dense R@5, points, SQuAD / MuSiQue. The gap is to E0's raw frozen read (87.69 / 45.72). The calls are on won
  back against the arm's first encoder raw (report.json has the intervals).
- **Both fused arms ADVANCE, on GNX's qwen3e06 bars (-4.25 / -1.25).**
  - On SQuAD they clear the bar by 0.08 and 0.23 points; on MuSiQue they clear it by far.
  - Neither clears zrckpn's SQuAD bar (-3.79).
  - The bars are pointers: they are the share of the gap qwen3e06's models won back, not these arms' models.
- **MuSiQue: both fused one-vector arms are above frozen.**
  - Against E0's raw frozen read: +3.28 and +2.43.
  - Against frozen's own V from E2B's frozen grid (87.75 / 47.63): +1.37 [+0.52, +2.22] and +0.52 [-0.37, +1.35].
  - Fusion is what does it: leaving w out costs 3.82 and 2.97 points on MuSiQue (0.60 and 0.75 on SQuAD).
  - R@1000 on MuSiQue: 86.81 and 86.29, against frozen's 87.25.
- **SQuAD: the one-vector gap stays near 4 points (order).**
  - With windows (F), every arm with gtemodern is level with or above frozen's raw read: +0.45, +1.01 and +1.24.
  - Against frozen's own F (90.45 / 48.57), F trails on SQuAD by 2.30 (gtemodern), 1.74 and 1.52 (the fused arms).
  - On MuSiQue, F is level with frozen's F for both fused arms (+0.52 and +0.06).
- **Against qwen3e06's recipes (E2B's grid):** every gtemodern arm is above qwen3e06's V and F on both carves. For
  example, gtemodern+granitesm V is +1.67 / +5.65 above qwen3e06 V.
- **Cost:**
  - gtemodern+granitesm runs two encoders per question (149M + 48M), and its one vector is 1,152 wide, under frozen's
    1,536.
  - gtemodern+granite runs 149M + 149M at 1,536.
- **Disclosed:**
  - As in E2B, G1 checks each arm's first encoder only. granite's raw read inside its fused arm is not G1-checked. Its E0
    read passed GR in 17.8.
  - The frozen V and F references are E2B's grid reads, taken before E2B's report.
- **Next (18.7):**
  - The best arm is gtemodern+granite (mean V gap -0.45).
  - gtemodern+granitesm (mean -0.80) is within 0.5 points with fewer parameters, so it is named too.
  - Their E2 stage would be 12.9's world on V's vectors: GNX and zrckpn at seed 0, trained on SQuAD fit and MuSiQue
    s1fit, read on the two s1eval carves against the frozen world. It is put to the user before it is declared.
  - Nothing was trained. The user decides D101.
