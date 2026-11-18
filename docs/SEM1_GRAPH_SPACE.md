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
- **The self-test.** `zsem1.py selftest --real` passed on the laptop after this page was committed (84772ea). It
  checked:
  - the space against a dense formula on a toy graph with an isolated node, for all eight settings;
  - RRF against `zenc.rrf_top` with two lists and against a dense float32 sum with three, with ties;
  - the in-memory search against `zenc.search`: two full shards and a part, duplicated rows, 1,500 questions;
  - the real SQuAD graph with the frozen vectors: 989,371 pairs, 0 isolated, rows summing to 1, unit norms, and the
    search on 20 question rows equal to `zenc.search`.

  No gold was read and no recall was computed.

## 11. Prior work (added 18 Nov after the declaration; it changes no rule above)

The mechanism is not new. Averaging a vector with its graph neighbours, offline and without training, is known from:
- **Retrofitting** (Faruqui et al., 2015): word vectors pulled toward lexicon neighbours, post hoc;
- **SGC** (Wu et al., 2019) and **NAFS** (Zhang et al., 2022): precomputed, parameter-free feature propagation for node
  classification;
- **Database-side augmentation in image retrieval** (Turcot and Lowe, 2009; Arandjelovic and Zisserman, 2012; the alpha
  form of Radenovic et al., 2018): each indexed vector replaced offline by a weighted mean of itself and its k nearest
  neighbours. This is SEM1's form, with neighbours taken from the same embedding;
- **Score regularization** (Diaz, 2005 and 2007): scores, not vectors, smoothed over a document-similarity graph at query
  time.

Other work uses the graph in two other ways:
- **At query time:** GAR (MacAvaney et al., 2022) and LADR (Kulkarni et al., 2023) walk a kNN corpus graph, and
  HippoRAG 2 (Gutierrez et al., 2025) runs PPR over an LLM-built graph.
- **Trained into the encoder:** hyperlink pre-training (HLP, Zhou et al., 2022; LinkBERT, Yasunaga et al., 2022),
  citation graphs (SPECTER, SciNCL), neighbour context (Contextual Document Embeddings, Morris and Rush, 2024), and GASE
  (Ding et al., 2026: frozen LLM embeddings with structure injected into a subspace, for node classification).

**What was not found** (18 Nov searches; not exhaustive):
- an encoder-free, LLM-free, rule-built and typed link graph baked into a passage index as a first-stage channel for
  multi-hop retrieval, so that one vector lookup carries the links' reach;
- any test of whether such a graph-in-the-index closes the gap between a small and a large encoder.

**What a later stage must add.** The kNN graph from the arm's own vectors (database-side augmentation's form) is the
baseline a reader would ask for. SEM1 as declared does not read it. It is named here for a later stage and is not
declared. Noisy graphs are known to hurt retrofitted vectors. MuSiQue's largest degree (14,300) is the case to watch.

## 12. Results (filed 18 November 2026, 20:16 IST): Z1, Z2 and Z3 NOT_HELD; the space closes as a first-stage channel

`zsem1.py report` on the host (`outputs/sem1/report.json` and `report.md`). Every record carries script sha256
5f0447b3... (ed4e426). All 25 rows ran: 12 sel, 12 read, one report.

**Gates.** G1, G3 and G4 held. G2 held exactly on all 12 reads: the recomputed base hits equal E0's `h_rrf5` on 100% of
questions, and the base R@5 equals E0's on every read.

**Chosen settings (s1sel).** frozen h2, a = 0.1; qwen3e06 h1, a = 0.25; mpnet h1, a = 0.1; mpnetqa h2, a = 0.1. Each
arm's choice sits at the low end of the ladder, and s1sel's mean add R@5 falls as a grows (frozen: 0.4654 at a = 0.1
down to 0.4493 at a = 1.0). Less graph scored better on the selection carve itself.

**The calls** (s1eval; points, 95% intervals, paired bootstrap over questions):

| dataset | arm | base R@5 | C1 add − base | C2 add − ctrl | C3 add − frozen base | u − v, R@1000 |
| --- | --- | ---: | --- | --- | --- | --- |
| MuSiQue | frozen | 47.34 | +0.27 [−0.19, +0.72] LEVEL | −0.03 [−0.22, +0.17] LEVEL | | +0.23 [+0.03, +0.43] ABOVE |
| MuSiQue | qwen3e06 | 44.98 | −0.09 [−0.48, +0.31] LEVEL | +0.67 [+0.41, +0.92] ABOVE | −2.44 [−3.15, −1.73] BELOW | +0.57 [+0.27, +0.88] ABOVE |
| MuSiQue | mpnet | 41.24 | −1.44 [−1.86, −1.00] BELOW | +0.11 [−0.06, +0.28] LEVEL | −7.54 [−8.40, −6.65] BELOW | +0.60 [+0.37, +0.84] ABOVE |
| MuSiQue | mpnetqa | 40.92 | −1.51 [−1.94, −1.06] BELOW | −0.01 [−0.21, +0.19] LEVEL | −7.93 [−8.80, −7.03] BELOW | +0.80 [+0.56, +1.05] ABOVE |
| SQuAD | frozen | 90.53 | −0.28 [−0.52, −0.04] BELOW | −0.04 [−0.14, +0.06] LEVEL | | −0.01 [−0.03, +0.02] LEVEL |
| SQuAD | qwen3e06 | 87.17 | −1.51 [−1.80, −1.23] BELOW | −0.22 [−0.40, −0.05] BELOW | −4.87 [−5.33, −4.43] BELOW | −0.09 [−0.18, −0.01] BELOW |
| SQuAD | mpnet | 83.98 | −2.36 [−2.70, −2.04] BELOW | −0.03 [−0.16, +0.08] LEVEL | −8.91 [−9.49, −8.36] BELOW | −0.02 [−0.07, +0.03] LEVEL |
| SQuAD | mpnetqa | 85.02 | −2.16 [−2.46, −1.84] BELOW | −0.11 [−0.24, +0.01] LEVEL | −7.67 [−8.19, −7.13] BELOW | +0.01 [−0.04, +0.06] LEVEL |
| MetaQA | frozen | 0.47 | +0.28 [+0.19, +0.38] ABOVE | −0.00 [−0.04, +0.02] LEVEL | | +0.94 [+0.78, +1.13] ABOVE |
| MetaQA | qwen3e06 | 0.35 | +0.03 [−0.01, +0.09] LEVEL | −0.00 [−0.03, +0.02] LEVEL | −0.09 [−0.18, +0.01] LEVEL | +2.03 [+1.77, +2.31] ABOVE |
| MetaQA | mpnet | 0.54 | +0.09 [+0.02, +0.16] ABOVE | −0.02 [−0.06, +0.00] LEVEL | +0.16 [+0.04, +0.27] ABOVE | +0.25 [+0.10, +0.40] ABOVE |
| MetaQA | mpnetqa | 0.46 | +0.02 [−0.03, +0.06] LEVEL | −0.02 [−0.05, +0.01] LEVEL | −0.00 [−0.10, +0.09] LEVEL | +0.25 [+0.12, +0.39] ABOVE |

**Expectations.** Z1 is NOT_HELD for every arm (no MuSiQue read has C1 and C2 both ABOVE). Z2 is NOT_HELD: qwen3e06's
C3 on MuSiQue is −2.44, against E0's −2.36. Z3 is NOT_HELD: every arm's C1 on SQuAD is BELOW.

**What the reads say.**
- **The graph adds nothing at the top that a second copy of the dense list does not.** C2 is LEVEL on 10 of 12 reads.
  It is ABOVE once (qwen3e06 on MuSiQue, where ctrl falls below base and add stays at base) and BELOW once (qwen3e06 on
  SQuAD).
- **A third list hurts where the dense list is weak.** For mpnet and mpnetqa, add is 1.4 to 2.4 points BELOW base on
  MuSiQue and SQuAD. The smoothed list pulls the fused order toward the weaker dense side, as ctrl does.
- **Little is won back.** Of the golds each small arm loses against the frozen base's top 5, add wins back 1.5% to 10%
  (MuSiQue, qwen3e06: 35 of 349). It drops 91 to 340 of the arm's own base top-5 golds.
- **The smoothed space reaches deeper, not higher.** u's R@1000 is ABOVE v's on MuSiQue and MetaQA for every arm
  (MuSiQue +0.23 to +0.80, MetaQA +0.25 to +2.03), and LEVEL or just BELOW on SQuAD. The golds it adds sit far below
  the top 5.
- MetaQA's first-stage R@5 is under 1% for every arm (E0's), so its two ABOVE C1 calls sit on a floor.

**D98.** Section 8's rule applies. Z1 failed for every arm, so the space as a first-stage channel closes, beside P5a to
P5c. The gain-back question goes to E2 (the models on each arm's pools). Before this result, on 18 November, the user
read it the same way: "smoothing the original doc vector with neighbours i do not think it would work, because we are
limited by the dimentions". The user's next form is a learned one: edges as functions of any frozen vectors, with
attention choosing which links to follow. It is sketched in [SEM2_EDGE_OPERATORS.md](SEM2_EDGE_OPERATORS.md), and it
will be declared there before any number. Nothing here is adopted.
