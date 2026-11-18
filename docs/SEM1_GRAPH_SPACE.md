# SEM1: a graph-smoothed document space as a first-stage channel, on the small and hard datasets (declared 18 Nov 2026, before any SEM1 number)

Code: `outputs/mp_unified/zsem1.py`, new. It imports `zenc.py`, `zenc1.py`, `zul1.py` and `zap1.py` and edits none of
them. Outputs go only under `outputs/sem1/`, and its arrays stay uncommitted. CPU only: nothing trains, and no model,
pool, fit or record changes.

## 1. Why

- **The user's direction (18 Nov).** "check only the small and hard datasets and see what do we lose and can we gain back
  etc which will help us in getting a better encoder, dont waste space nor time". The user also said "Yes, start it" to
  starting the semantic-space work on the CPU before the encoder is final.
- **ENC's small-dataset results** (`docs/ENC_ENCODER_COMPARISON.md`, section 8). On MuSiQue, a smaller encoder loses 2.4
  to 6.4 points of RRF R@5 at the first stage. It loses almost nothing in links: the union of S, R and the arm's K stays
  level or above. The loss to win back is in the first stage.
- **The form named but never run** (`docs/PROGRAM_2026_10_13.md`, lines 151-153: the space's nearest neighbours of the question as a channel beside dense and SPLADE, against the encoder gap). The space's nearest neighbours of the
  question, used as a first-stage channel. P5a to P5c read a learned or offset node space inside the models. None of
  them used the space as a first-stage list, and none ran on the S/R/K layout.

## 2. Arms and datasets

- **Arms.** frozen (gte-Qwen2-1.5B), qwen3e06, mpnet, mpnetqa: ENC's four, with the vectors E0 searched.
- **Datasets.**
  - **MuSiQue:** the hard multi-hop set. The expectations are read here.
  - **SQuAD:** the single-hop control. The space should do no harm there.
  - **MetaQA:** read beside. Its golds are answer sets, and every arm's RRF R@5 is under 1%.
- **Not read.** HotpotQA, 2Wiki and WebQSP: the user held their encodes (`hold-enc-big`).

## 3. The graph (encoder-free, the same for every arm)

G_d = W2_S ∪ F0T, plus S on MetaQA (A1's domains: S only where the data has structure). These are the pairs of D83's
layout without K. They come from `zul1.families`, which checks each family against its build record (UL1's gate G1).

- **Why K is left out.** K is built from an encoder's vectors. Leaving it out keeps the graph identical across arms, so
  any difference between arms comes from the vectors alone.
- **Edge labels are not read.** Nothing here is learned, so every edge counts the same. The labels (S.<rel>, R.*) go
  into the learned form named in section 8, as the user's rule asks.
- **Graph size** (pairs; probed on the laptop before this page, as recorded in the development log):

  | dataset | pairs | isolated nodes | median degree | largest degree |
  |---|---|---|---|---|
  | SQuAD | 989,371 | 0 | 74 | 991 |
  | MuSiQue | 3,209,654 | 3,436 | 25 | 14,300 |
  | MetaQA | 257,746 | 0 | 5 | 5,098 |

## 4. The space

- **Inputs.** V is the arm's document vectors, each row scaled to unit length. A is the row mean over a node's
  neighbours in G_d: unweighted and symmetric. An isolated node's row of AV is zero.
- **One hop (h = 1):** U = unit(V + α·AV).
- **Two hops (h = 2):** U = unit(V + α·AV + α²·A(AV)).
- **The ladder:** α ∈ {0.1, 0.25, 0.5, 1.0} × h ∈ {1, 2}, eight settings.
- **The question.** The question vector is the arm's own, unchanged. It is scored against U by an exact float32 inner
  product over the whole corpus, top 1,000, ties to the lower node id, as E0's search does.

## 5. The first-stage forms

**The lists.** For each question, three top-1,000 lists:
- **v:** the arm's own dense list, E0's search recomputed;
- **u:** the space's list;
- **SPLADE:** the record's list, unchanged.

**The forms.** Each is a float32 RRF over its lists: 1/(60 + rank) per list, first occurrence only, summed in the order
listed, then ordered by score and then by node id.
- **base** = RRF(v, SPLADE): E0's RRF R@5.
- **swap** = RRF(u, SPLADE).
- **add** = RRF(v, u, SPLADE).
- **ctrl** = RRF(v, v, SPLADE). This is add with the graph taken out (the dense list counted twice). It separates the
  graph's part from the extra weight a third list gives.

**Recall.** Per question, min(golds in the top k, golds) / golds, averaged over the questions that have a gold.

## 6. Selection and read (two carves, no peeking)

- **Selection, on s1sel only** (UL1's carve: `zul1.carve_questions(d, "s1sel")`, checked against `carves.json`). For each
  arm and dataset, the eight settings' add R@5 is computed, with swap R@5 and u-dense R@5 beside. The setting chosen is
  the one with the highest mean add R@5 over the three datasets: one setting per arm, used on all three. Ties go to the
  smaller α, then to h = 1. MetaQA's s1sel values are near zero, so in practice the choice is made on MuSiQue and SQuAD.
- **Read, on s1eval** (E0's questions and golds: `zenc.s1eval` and `zenc.gold_sets`). Only the chosen setting is read.

**Columns:**
- u-dense R@5, R@20 and R@1000;
- base R@5, swap R@5, add R@5 and R@20, ctrl R@5.

**Calls.** Each is a paired difference per question with `zap1.boot_diff` (2,000 resamples of the questions, seed 0,
one slot per question), called ABOVE, LEVEL or BELOW at 95%.
- **C1:** add minus base, the arm's own.
- **C2:** add minus ctrl, the graph's own part.
- **C3:** a small arm's add minus the frozen arm's base. This is how much of the encoder's loss comes back.

**Beside:**
- swap minus base;
- u-dense R@1000 minus v-dense R@1000;
- **lost and won back.** For each small arm, a lost gold is a (question, gold) that the frozen base top 5 holds and the
  arm's base top 5 does not. The column gives the share of lost golds that the arm's add top 5 holds. It also gives the
  count of golds in the arm's base top 5 that add drops, which is the harm.

## 7. Gates (each stops the row)

- **G1.** The graph: `zul1.families`' own checks. The node count must equal the arm's document rows.
- **G2.** The base reproduces E0. On s1eval, the arm's recomputed base hits must equal E0's per-question `h_rrf5`
  (`outputs/enc/e0/<arm>/<ds>.npz`, the same ids in order) on at least 99.5% of questions. Its R@5 must be within
  0.0005 of E0's.
- **G3.** The vectors: the shapes match the manifest and the encoder's dimension, and nothing is NaN. U's rows must be
  unit length within 1e-4.
- **G4.** `zsem1.py selftest`:
  - on a toy corpus, U against a dense-matrix formula;
  - RRF against a brute-force sum, with ties;
  - the in-memory search against `zenc.search` on the same shards;
  - on the laptop, the real SQuAD graph with the frozen vectors: A's rows sum to 1, isolated rows equal unit(V), and
    U's norms. No gold is read.

## 8. What SEM1 decides

Nothing is adopted. Three expectations, each called HELD or NOT_HELD:

- **Z1.** On MuSiQue, for each of the four arms, C1 is ABOVE and C2 is ABOVE. The space adds to the first stage, and the
  gain comes from the graph, not from the extra list.
- **Z2.** On MuSiQue, qwen3e06's C3 is LEVEL or ABOVE. The space wins back the whole first-stage loss of the closest
  arm (E0: -2.36 points).
- **Z3.** On SQuAD, no arm's C1 is BELOW.

**D98** (open; it is the user's call). If Z1 holds for an arm, the next form is declared on this page before its numbers.
There are two candidates:
- (a) the space's list in the first stage of D83's pools for that arm (E1b and E2 on these datasets);
- (b) a learned form that reads the edge labels.

If Z1 fails for every arm, the space as a first-stage channel closes, beside P5a to P5c. The gain-back question then
goes to E2: the models on each arm's pools.

## 9. Rows

All rows run on the host CPU with 8 BLAS threads.
- **`sem1-sel-<arm>-<ds>`** (12 rows). Each runs after `enc-first-<arm>-<ds>` and writes
  `outputs/sem1/sel/<arm>/<ds>.json`. Memory is 8 GB on MuSiQue and 4 GB on the others.
- **`sem1-read-<arm>-<ds>`** (12 rows). Each runs after its arm's three `sel` rows. It writes
  `outputs/sem1/read/<arm>/<ds>.json`, and an `.npz` holding the ids and the base, add and swap top 5 per question.
- **`sem1-report`.** It runs after the 12 reads and writes `outputs/sem1/report.json` and `report.md`.
- **Then the laptop:** fetch, then a results section on this page.

## 10. Development log (disclosed before any SEM1 number)

- **The graph probe.** Before this page, a laptop probe built G_d for the three datasets with `zul1.families`. It printed
  only the sizes in section 3 and the shard counts. No vector was searched and no gold was read.
- **The self-test.** `zsem1.py selftest` runs on the laptop after this page is committed. Its results are added here
  before any row runs.

## 11. Results

(Not yet filed.)
