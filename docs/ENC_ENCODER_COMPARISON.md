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
