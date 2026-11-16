# ENC: smaller encoders against the frozen one (D93), stage E0: the first stage

Declared 17 Nov 2026, 04:15 IST, before any number. Code: `outputs/mp_unified/zenc.py`. Records: `outputs/enc/`.

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

(Not yet filed.)
