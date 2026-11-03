# PFS: what PKO's cached inputs and pool edges carry, read on the CPU (31 October 2026)

Declared 31 October 2026, before any number of the stage. Code: `outputs/mp_unified/zpfs.py` (self-test passes). Outputs only under
`outputs/pfs/`. No model is fit and no sealed folder is written. Nothing here is a result.

## 1. Why

The user, 31 October: "use the cpu to find the best possible features to give with these new pools so that we can have the best possible
ranking to be exploited, we are missing something there, we should parallely do these quick tests to find and then train the models at the
same time". And, while this was being built: "improving the gnn is the priority, mlp follows close behind and gnn should not be built over
mlp it needs to be the best model we can get in the gnn".

What is known (filed, not new):
- **The GNN of record is zg6r**, its own model (G6, docs/G6_FREE_GNN.md: zrm's block encoder as its input stage, three hidden-state layers
  over three edge families, an unbounded readout, no MLP score in the forward). zspk is the MLP plus one propagation step and is the lean
  baseline, not the GNN. On PKO's pools, ML1's seed-0 reads of the WQ-musique split (`scr-ml1-*-s0/read.json`, p@swa R@5; not a filed
  verdict, one seed): hotpotqa zg6r 0.859, zrckp 0.842, zspk 0.844; 2wiki 0.927, 0.905, 0.913; webqsp 0.663, 0.656, 0.640; squad 0.914 for
  all three; musique (zero-shot) 0.509, 0.505, 0.488; metaqa (zero-shot) 0.362, 0.399, 0.356. The GNN leads the MLP in domain by about two
  points on hotpotqa and 2wiki and is below it on metaqa. Reach on hotpotqa and 2wiki is about 0.96, so the room is in the ranking.
- **The ranker lacks information, not capacity** (docs/DIAG_BRIDGE.md, zlink's docstring): the golds a ranker misses in partly-found questions
  sit next to a found gold (within two hops: 2wiki 0.77, hotpotqa 0.92-0.93, musique 0.38 and 0.77; lift 6 to 14); a stronger scorer on the same
  per-row inputs gains nothing (FEATURE_LIMIT). The pool's edges are where that information is; a GNN is the model that can use it.
- **The model already normalises across the pool.** Every lean block enters as raw and as a within-question z-score (`seg_zscore`), and
  lean_mlp8's context adds the pool's mean, sd and max per column. Pool-relative rescaling of the current columns is not a missing feature.
  (The SEMB z-score ruling the user owes concerns SEMB's learned form, not this.)
- **zg6's attention logit reads the sender and the question's context only** (l = <W_a h_u, c> / 8 + beta_f). No per-edge value, and nothing of the
  receiver, enters it. The links hold only (u, v, family).

So two questions can be asked without a model, on the CPU, in minutes, before a GPU fit is spent on either: which cached columns can separate a
gold from the rows beyond the first stage's top five, and how much do edge-made scalars add to node-only scalars when the question is which
neighbour of a top row is the missing gold.

## 2. The two reads

Both on each dataset's PKO fit carve (musique: its `s1fit` carve; metaqa has no fit carve and is not read), the first `--nq` questions in the
carve's order (2wiki 1,000; hotpotqa 1,000; squad 1,000; webqsp 600; musique 400), basis hotpotqa, on the host's CPU (6 threads, one item per
read). Labels describe the data and are never an input to anything here.

**PFS-A, `columns`.** For each of the lean pick's cached columns (rank: dense_rr, splade_rr, rrf, agreement, is_seed; dense_cos; topo_STRUCT;
depth_STRUCT; WALK; WALKF; SEED; DISTS): the modal value and the share of rows at it, the gold rate at the mode and elsewhere, the share of golds
at the mode, and the AUC of the column's within-question z-score for gold, over all rows (a random 1,000,000 of them, seed 0) and over the rows
beyond the first stage's top five (rrf rank >= 5 inside the pool), where the golds a ranker misses are.

**PFS-B, `edges`.** First stage = the pool's top K by rrf (K = 5 and 20). A *receiver* is a row outside the top K with an edge from a top-K
row. Per dataset and K:
- one-hop reach: the share of the golds outside the top K that are receivers, overall and per edge family (0 structural, 1 NER, 2 kNN, as the
  links store them);
- among receivers, per scalar (the first stage's z-score, dense_cos, SEED's first column, the largest cosine of the stored 128-wide projections of
  the receiver and a top-K sender, the rank-weighted count of top-K senders, minus log of the receiver's degree): the AUC for gold and the share of
  questions (with a gold receiver) whose top 5 / top 20 receivers hold a gold, beside the exact share a random order gives;
- a logistic fit on the node-only scalars (first-stage z, dense_cos, SEED's three columns) and one on node plus edge scalars (the cosine, the log
  sender count, the rank-weighted sender count, the log degree, the families on the pair), each cross-fitted on question parity (fit on one half,
  scored on the other, both ways), with the same AUC and hits.

## 3. Reading, fixed before the numbers

PFS-A, per dataset (fixed thresholds):
- **SPARSE(column)**: the share at the mode >= 0.5.
- **CARRIES(column)**: the column's AUC beyond the top five >= 0.60. **NO_CARRIER(dataset)**: no column reaches 0.60 there.
- **What follows.** NO_CARRIER on a dataset: no cached column separates the misses on it, so the missing information is not in the cache, and the
  next stages add information (an edge score, a lexical score for every pool row), each its own declared file. CARRIES on a column: it is where the
  separation is, and a block ablation (drop-one) on that column is the next screen. A column that is SPARSE and has gold rate at the mode below
  the rate elsewhere is a coverage hole for the rows it misses, reported by name.

PFS-B, per dataset and K:
- **ONE_HOP_REACHES**: one-hop reach >= 0.60 of the golds outside the top K. (DIAG_BRIDGE's number, re-read on PKO's pools.)
- **EDGE_ADDS**: the node-plus-edge fit's hit@5 exceeds the node-only fit's by >= 0.03 *and* its AUC exceeds by >= 0.01. **EDGE_ADDS_NOTHING**:
  both differences are within 0.01. Otherwise MIXED, reported with the numbers.
- **What follows.** EDGE_ADDS on at least three of the four datasets with edges (2wiki, hotpotqa, musique, webqsp), at K = 5: the GNN stage U2u
  (the link score as an input to the edges, stage 8 of docs/PROGRAM_2026_10_30.md) is declared with the scalar that carries (the largest
  standardised weight of the node-plus-edge fit named) in zg6's attention logit and message gate, as its own file, to run on the GNN of record
  (zg6r, the host's GPU or Kaggle's T4). EDGE_ADDS_NOTHING on at least three: these scalars are closed for the GNN's edges, and the GNN's room is
  on the node side (PFS-A's carriers and holes) or in a different edge quantity. Anything else: reported, and no stage is declared from it.

## 4. What this is not (stated so a reader does not credit it)

- **Not a model result.** The first stage is rrf's top K, not a trained ranker's. ML1's fits find more than rrf does (hotpotqa R@5 0.86 against
  0.69), so the "missing" golds here include many a model already finds; K = 20 stands for a decent ranker's found set. The receivers' hit@m is a
  hit over the receiver set, not a final R@5.
- **Not the lean MLP and not zg6r.** The logistic fits are small probes of what the scalars carry. A gain here is a reason to declare a GPU
  screen, not a gain.
- **Fit carves, first questions only.** These are training questions in the carve's order; the numbers describe the data, and a different slice
  would give other figures. There is no seed here; the logistic fits are cross-fitted on question parity.
- **The edge scalars are the ones the cache can make.** The links hold (u, v, family). A link rule's own score (title-mention counts, the
  sentence the mention came from) is not stored and is not read here; that would be a build, declared if this reading calls for it.
- **Not the user's MLP-first features.** The MLP-trainer screens (drop-one, add-one on lean_gpu's CPU path) are not in this file. If PFS-A names
  carriers or holes, they are declared as their own file with their own base and seed null, since a new device needs its own (the 13 October rule).

## 5. Files and items

    outputs/mp_unified/zpfs.py            columns, edges, --selftest
    outputs/pfs/columns_<dataset>.json    PFS-A, five datasets
    outputs/pfs/edges_<dataset>.json      PFS-B, four datasets (squad has no passage graph worth reading; it is read too, and reported as it is)

Feeder items `pfs-cols-<dataset>` and `pfs-edges-<dataset>`, CPU only, after PKO's gate; they read PKO's cache and links and write only under
`outputs/pfs/`. They hold PKO's cache pinned for the cleaner while queued; ML1's GPU items hold it beyond them.

## 6. Results

Filed as section 8, below Amendment A1 (section 7), which came between the reads and this filing.

## 7. Amendment A1, 31 October 2026, after PFS-B's first numbers were seen and before any number of the reading it adds

**What was seen.** PFS-B's reads ran in section 2's form (the node-only baseline of five scalars). Against it, the node-plus-edge fit adds to hit@5 at
K = 5: hotpotqa +0.32, 2wiki +0.57, webqsp +0.35 (AUC +0.13, +0.21, +0.32); squad -0.06 (AUC -0.01); musique had not finished. By section 3's rule that
is EDGE_ADDS on three of the four datasets with edges. It is a weak comparison: the five scalars leave out the lean pick's own seed-walk (WALK, WALKF),
topology and distance columns, which already count a row's adjacency to the first stage, so the first reading overstates what edges add to a model
that has them. The largest weights were the rank-weighted sender count, the receiver's degree and the structural family, the same quantities
WALK holds.

**What A1 adds** (`edges --full`, `edges_full_<dataset>.json`; the reads of section 2 stand as they ran): a second baseline of all 86 cached columns of the
lean pick, raw and as the lean models see them (z-scored over the whole pool), 172 columns, fitted as before (logistic, cross-fitted on question parity);
and the same fit plus the seven edge scalars (cosine, log sender count, rank-weighted sender count, log degree, three families).

**Rule, fixed here.** Per dataset at K = 5, hit@5 and AUC of the receivers:
- **EDGE_ADDS_OVER_LEAN**: the full-plus-edge fit exceeds the full fit by >= 0.03 in hit@5 and >= 0.01 in AUC.
- **LEAN_HAS_IT**: both differences within 0.01. Otherwise MIXED, reported with the numbers.
- **What follows.** Section 3's "what follows" for the edges now reads: U2u (stage 8) is declared, on the GNN of record, only on EDGE_ADDS_OVER_LEAN on at
  least three of the four datasets with edges (2wiki, hotpotqa, musique, webqsp), naming the edge scalar with the largest standardised weight in the
  full-plus-edge fit. The first reading (EDGE_ADDS against five scalars) is reported as it ran and declares nothing by itself. LEAN_HAS_IT on three: these
  scalars are in the lean columns already, the U2u form on them is not declared, and the next form is an edge quantity the cache does not hold (the link
  rule's own score, a build declared as its own file).
- **Stated against it.** A logistic fit on 172 columns can miss what an MLP finds, so LEAN_HAS_IT is weak evidence of redundancy and EDGE_ADDS_OVER_LEAN is the
  stronger statement. Neither is a model result.

## 8. Results (filed 31 October 2026; the reads ran on the host's CPU, the first questions of each fit carve, no seed)

**PFS-A, the columns.** Beyond-top-5 AUC is the within-question z-score of a column for gold over the rows the first stage ranks 5th or lower; CARRIES is >= 0.60.

| dataset | questions / rows | golds beyond rrf top 5 | columns CARRYING (of 86) | NO_CARRIER | SPARSE (mode share >= 0.5) | SPARSE with gold rate at the mode below the rate elsewhere |
|---|---|---|---|---|---|---|
| hotpotqa | 1,000 / 711,780 | 576 | 70 | no | 48 | 38 |
| 2wiki | 1,000 / 1,073,464 | 980 | 58 | no | 50 | 40 |
| webqsp | 600 / 2,922,458 | 2,977 | 62 | no | 57 | 48 |
| musique | 400 / 2,221,400 | 306 | 53 | no | 45 | 35 |
| squad | 1,000 / 50,031 | 62 | 23 | no | 45 | 24 |

- **Where the separation is differs by dataset.** On hotpotqa, 2wiki and webqsp the carriers are the graph columns (topo_STRUCT, depth_STRUCT, WALK, WALKF, DISTS; best beyond-top-5 AUC 0.89, 0.95, 0.91).
  On 2wiki and webqsp the first stage's own columns do not carry there (dense_rr 0.38 and 0.55, splade_rr 0.56 and 0.41, rrf 0.42 and 0.51): the misses are separated by the graph and by nothing the first
  stage knows. On musique and squad it is the other way: the first-stage columns carry (musique rrf 0.92, dense_cos 0.91, splade_rr 0.87; squad dense_cos 0.89) and the graph columns top out lower
  (musique topo_STRUCT 0.72, WALK 0.72, depth_STRUCT 0.79, DISTS 0.79; squad no graph block above 0.65).
- **Caveat on the graph carriers' size.** The pool was built by walking the same links, so a row with a nonzero walk value is in the pool because it was reached; part of the 0.9+ is reachability, not
  ranking. The read cannot split the two.
- **Coverage holes (SPARSE, with the gold rate at the mode below the rate elsewhere), common to all five datasets:** `is_seed`, WALK[3] to WALK[8] and WALK[15], WALKF[12], [15], [16], [17]. These
  columns are zero for most rows and informative where they are not; they say nothing for the rows a walk did not reach.
- NO_CARRIER holds on no dataset: the cached columns do separate the misses. Under section 3 the next screen on PFS-A is the block ablation (drop-one) on the carrying blocks; it is not declared here.

**PFS-B, the edges**, in section 2's form (five node scalars against the same plus the edge scalars):

| dataset | K | golds outside top K | one-hop reach (fam 0 / 1 / 2) | receivers | node-only hit@5 (chance) | + edge scalars | d hit@5 | d AUC | call |
|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 5 | 576 | 0.724 (0.64 / 0.24 / 0.20) | 82,381 | 0.346 (0.087) | 0.667 | +0.321 | +0.126 | EDGE_ADDS |
| hotpotqa | 20 | 377 | 0.870 (0.80 / 0.28 / 0.23) | 152,670 | 0.178 (0.042) | 0.537 | +0.359 | +0.209 | EDGE_ADDS |
| 2wiki | 5 | 980 | 0.730 (0.71 / 0.26 / 0.11) | 93,240 | 0.231 (0.070) | 0.796 | +0.565 | +0.207 | EDGE_ADDS |
| 2wiki | 20 | 808 | 0.896 (0.87 / 0.33 / 0.13) | 178,009 | 0.196 (0.036) | 0.759 | +0.563 | +0.227 | EDGE_ADDS |
| webqsp | 5 | 2,977 | 0.519 (0.51 / 0.03 / 0.03) | 155,917 | 0.241 (0.124) | 0.594 | +0.353 | +0.318 | EDGE_ADDS |
| webqsp | 20 | 2,896 | 0.741 (0.73 / 0.05 / 0.03) | 312,803 | 0.144 (0.071) | 0.481 | +0.337 | +0.313 | EDGE_ADDS |
| musique | 5 | 306 | 0.507 (0.33 / 0.16 / 0.21) | 92,734 | 0.457 (0.034) | 0.457 | +0.000 | +0.012 | MIXED |
| musique | 20 | 175 | 0.623 (0.37 / 0.28 / 0.25) | 270,886 | 0.222 (0.011) | 0.185 | -0.037 | +0.020 | MIXED |
| squad | 5 | 62 | 0.758 (0.68 / 0.29 / 0.44) | 26,647 | 0.681 (0.182) | 0.617 | -0.064 | -0.005 | MIXED |
| squad | 20 | 13 | 0.846 (0.69 / 0.62 / 0.62) | 21,362 | 0.545 (0.189) | 0.545 | +0.000 | +0.005 | EDGE_ADDS_NOTHING |

- ONE_HOP_REACHES at K = 5: hotpotqa, 2wiki and squad yes; webqsp (0.519) and musique (0.507) no. At K = 20 all five reach 0.60 or more. Structural links (family 0) carry most of the one-hop reach on every
  dataset; on musique they reach only 0.33 to 0.37 of the missing golds, and the NER and kNN families 0.16 to 0.28.
- Read alone, this is EDGE_ADDS on three of the four datasets with edges (hotpotqa, 2wiki, webqsp), which section 3 would have turned into U2u. That is why A1 added the stronger baseline first.

**Amendment A1, the same receivers against all 86 lean columns (raw and pool-z-scored, 172 columns), with and without the seven edge scalars:**

| dataset | K | lean columns hit@5 / AUC | + edge scalars hit@5 / AUC | d hit@5 | d AUC | A1 call |
|---|---|---|---|---|---|---|
| hotpotqa | 5 | 0.736 / 0.927 | 0.749 / 0.927 | +0.012 | +0.000 | MIXED |
| hotpotqa | 20 | 0.656 / 0.926 | 0.666 / 0.928 | +0.009 | +0.002 | LEAN_HAS_IT |
| 2wiki | 5 | 0.867 / 0.958 | 0.870 / 0.959 | +0.003 | +0.001 | LEAN_HAS_IT |
| 2wiki | 20 | 0.857 / 0.962 | 0.865 / 0.964 | +0.008 | +0.002 | LEAN_HAS_IT |
| musique | 5 | 0.556 / 0.958 | 0.563 / 0.959 | +0.007 | +0.001 | LEAN_HAS_IT |
| musique | 20 | 0.380 / 0.959 | 0.380 / 0.961 | +0.000 | +0.002 | LEAN_HAS_IT |
| webqsp | 5 | 0.702 / 0.882 | 0.712 / 0.887 | +0.010 | +0.005 | MIXED |
| webqsp | 20 | 0.593 / 0.903 | 0.625 / 0.910 | +0.032 | +0.006 | MIXED |
| squad | 5 | 0.745 / 0.873 | 0.766 / 0.865 | +0.021 | -0.008 | MIXED |
| squad | 20 | 0.545 / 0.728 | 0.545 / 0.723 | +0.000 | -0.004 | LEAN_HAS_IT |

**Reading (the fixed rules, applied as written).**
- **EDGE_ADDS_OVER_LEAN holds on none of the four datasets with edges at K = 5.** U2u (stage 8) is therefore **not declared** on these scalars. The first reading's EDGE_ADDS on three datasets was the weak
  baseline's: the lean pick already holds the quantities the edge scalars make (WALK, WALKF, topology and distance columns), and against it the seven add at most +0.012 hit@5 and +0.005 AUC at K = 5.
- LEAN_HAS_IT holds on two of the four at K = 5 (2wiki, musique); the other two are MIXED by the thresholds (hotpotqa +0.012 hit@5 and +0.000 AUC; webqsp +0.010 and +0.005; both are outside the 0.01 band in hit@5 only,
  by 0.0024 and by 0.00003). The rule's "three" for closing the scalars is not met either, so closure is not claimed: **these edge scalars are neither declared into the GNN nor closed**; the evidence is that they add very little
  to a model that has the lean columns.
- What the reads leave standing: the cached columns do separate the missed golds (PFS-A, 53 to 70 carriers on four datasets), and a logistic probe on the cached columns alone reaches receiver hit@5 of 0.74
  (hotpotqa), 0.87 (2wiki), 0.56 (musique) and 0.70 (webqsp) at K = 5. In the sense these reads can test, nothing cached is missing.
- Squad is the one dataset where the first stage is the signal and the graph is not (section 4's note stands).

**Stated against these results** (section 4 stands): rrf's top K stands in for a ranker's first stage; the fits are logistic probes on the receivers, not a final R@5 and not the lean MLP or zg6r; a nonlinear
model may use the same columns better than the probe does, which is why LEAN_HAS_IT is weak evidence; the graph carriers include reachability; the first questions of each fit carve were read, once, with no seed.

**What is next, named.** Two inputs are not in the cache and so were not tested: the link rule's own per-edge score (what a hyperlink's anchor text carries over a title match) and a first-stage-relative
similarity (to the centroid of the top-K rows). Each is a build or a derived block, declared in its own file before its number.

## 9. Amendment A2, 31 October 2026, after section 8 was filed and before any number of the reading it adds

**Why.** Section 8 leaves two inputs the cache does not hold untested. One is the link rule's own per-edge score; its earlier forms on the lean pair ran with no signal (docs/DIAG_LINK_WORDS.md,
docs/A6E_LINK_WORDS_EDGES.md) and it needs a build, so it is not read here. The other is a row's similarity to the first stage's found rows, which the lean pick does not carry as a column (SEED measures
similarity to the question's seed entities, not to the top rows; the edge scalars of section 2 measure it only for a receiver of an edge). It is made from what the cache holds, before any row is scored, so
a model may take it as an input without message passing.

**The read** (`prf --dataset D`, `prf_<dataset>.json`; the same fit carves, the same first questions, every pool row): three columns made from each row's stored 128-wide projection, normalised:
- `prf5`, `prf20`: the cosine to the centroid of the top 5 and of the top 20 rows by rrf (the centroid is normalised);
- `maxcos5`: the largest cosine to another of the top 5 rows (a top row is not compared with itself).

Each is read raw and as a within-question z-score (six columns). Per dataset: the AUC of each column's z-score for gold over all rows and beyond the first stage's top 5; then two logistic probes over all pool
rows, cross-fitted on question parity as before: the 172 lean columns of section 7 (the baseline), and those plus the six. Metrics over every row of every question: AUC, recall@5 and recall@20 (the share of a
question's golds among its top m rows, averaged over questions with a gold), and hit@5 with its exact chance.

**Rule, fixed here.** Per dataset, the probe with the six columns against the baseline:
- **PRF_ADDS**: recall@5 gain >= 0.02 and AUC gain >= 0.005.
- **PRF_NOTHING**: the absolute recall@5 difference <= 0.01 and the absolute AUC difference <= 0.005. Otherwise MIXED, reported with the numbers.
- **What follows.** PRF_ADDS on at least three of the five datasets: a derived block of these columns is declared as an add-one screen on the GNN of record (zg6r) first and the MLP second, as its own file with its
  own base and null, since a new input is a model change under the 13 October rule. PRF_NOTHING on at least three: first-stage-relative similarity is closed as an input for the probe and for the models
  that have the lean columns, and the cached node side has no cheap derived input left to try. Anything else: reported, and nothing is declared from it.
- **Stated against it.** A logistic probe on all rows is far below the models (it has no interactions), so a gain here can be larger or smaller than a model's. The first questions of each fit carve are
  read once, with no seed. rrf's top rows stand in for a ranker's. The columns use only the stored projection, not the frozen encoder's full vectors.

Items: `pfs-smoke-prf` (hotpotqa, 40 questions) and `pfs-prf-<dataset>` for the five datasets, CPU only, after PKO's gate; they write only under `outputs/pfs/`.

## 10. Results of Amendment A2 (filed 31 October 2026; the same questions as section 8, every pool row, no seed)

| dataset | questions / rows | golds beyond rrf top 5 | beyond-top-5 AUC of prf5 / prf20 / maxcos5 (z-scored) | lean 172 columns: recall@5 / recall@20 / AUC | + the six: recall@5 / recall@20 / AUC | d recall@5 | d AUC | call |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 1000 / 711,780 | 576 | 0.541 / 0.537 / 0.634 | 0.8053 / 0.9229 / 0.9871 | 0.8083 / 0.9229 / 0.9873 | +0.0030 | +0.0002 | PRF_NOTHING |
| 2wiki | 1000 / 1,073,464 | 980 | 0.426 / 0.379 / 0.566 | 0.8085 / 0.9314 / 0.9857 | 0.8083 / 0.9310 / 0.9857 | -0.0002 | -0.0000 | PRF_NOTHING |
| webqsp | 600 / 2,922,458 | 2,977 | 0.499 / 0.472 / 0.512 | 0.2919 / 0.5575 / 0.9530 | 0.2958 / 0.5625 / 0.9548 | +0.0039 | +0.0018 | PRF_NOTHING |
| musique | 400 / 2,221,400 | 306 | 0.767 / 0.789 / 0.798 | 0.6500 / 0.8163 / 0.9877 | 0.6500 / 0.8187 / 0.9876 | +0.0000 | -0.0001 | PRF_NOTHING |
| squad | 1000 / 50,031 | 62 | 0.654 / 0.645 / 0.665 | 0.9518 / 0.9950 / 0.9871 | 0.9538 / 0.9940 / 0.9872 | +0.0020 | +0.0001 | PRF_NOTHING |

**Reading (section 9's rule, applied as written).** PRF_ADDS on 0 of the five datasets, PRF_NOTHING on 5. PRF_NOTHING holds on 5 of the five datasets (at least three): first-stage-relative similarity is **closed** as an input for the probe and for models that have the lean columns, and the cached node side has no cheap derived input left to try.

- The new columns carry where the first stage carries and not where the graph does. Beyond the top 5 their AUCs are 0.77 to 0.80 on musique and 0.65 to 0.67 on squad (the two datasets whose
  first-stage columns carry, section 8), and 0.54 / 0.54 / 0.63 on hotpotqa, 0.43 / 0.38 / 0.57 on 2wiki and 0.50 / 0.47 / 0.51 on webqsp (prf5 / prf20 / maxcos5). Where they do carry they
  restate what dense_cos and rrf already give: the probe's recall@5 on musique is 0.650 with and without them. On 2wiki, where the missed golds are bridge passages, similarity to the found rows points the
  wrong way (prf20 below 0.5), as it should if the missing gold is reached by a link and is not a near neighbour of what was found.
- **The baseline probe is itself strong.** A logistic fit on the 172 cached columns, cross-fitted by question parity, puts recall@5 at about 0.81 on hotpotqa and 2wiki, 0.65 on musique, 0.95 on squad and 0.29 on webqsp,
  with AUC 0.95 to 0.99 over every pool row. These are the first questions of each fit carve, an out-of-fold probe, not the models' protocol, and are not compared with ML1's reads here.
- **Stated against these results** (section 9 stands): a probe without interactions; the stored 128-wide projection, not the encoder's vectors; the first questions of each fit carve; no seed.

## 11. Amendment A3, 3 November 2026, after LTA's first column records were seen and before any number of the reading it adds

**Why.** Section 10 closed one kind of similarity to the found rows: question-free, from the stored 128-wide projection. It left two things open, the second named in its own "stated against". One is a
similarity that depends on the question. The other is the frozen encoder's full vectors. LTA's first column records (`outputs/lta/cols`, hotpotqa, not filed) rank a gold that sits outside rrf's top five
in the list of its first top-five gold. Where that gold is not linked to the anchor, the two question-conditional free scores of LTW (`ltwdiag.pair_feats`) put it in the top two of that list more often
than rrf does, and more often than LTA's learned bil scorer. bil is link-dominated, and on unlinked targets it ranks below rrf, pair and res. Those records use an oracle anchor (a gold). The question is
whether the same scores carry against rrf's own top five, with no gold in sight, over every pool row, and beyond the 172 lean columns. A model may take them as an input. They are made from the frozen
vectors of the first stage's fixed top five before any row is scored. That is a fixed beam's read, not message passing (the 13 October rule; LTC section 2).

**The read** (`pair --dataset D`, `pair_<dataset>.json`; section 9's fit carves and first questions, every pool row): for a row c and each of rrf's top five rows a_k (unit vectors of the frozen encoder;
a top row is not paired with itself):
- `res_k = c.q - (a_k.q)(a_k.c)`, the part of q along c once a_k's direction is taken out;
- `pair_k = (a_k.q + c.q) / |a_k + c|`, q's cosine to the normalised sum of the pair.

Six columns are made from these: `pair_max`, `res_max`, `pair_mean` and `res_mean` over the five anchors, and `pair_1`, `res_1` against rrf's first row (its second row for the first row itself). Each is
read raw and as a within-question z-score, twelve columns in all. Each row's c.q must match the cache's dense_cos within chndiag's tolerance (0.002) on all but 1% of rows (the identity), or the read
refuses. That check makes the rows, node vectors and question vectors the cache's own. Section 9's probe and metrics are used as written: the 172 lean columns as the baseline, then those plus the
twelve. The baseline's recall@5 should reproduce section 10's within 0.002 (the same questions and columns). On musique, recall@5 is also reported by hop. Code: `outputs/mp_unified/zpfp.py`, which
imports zpfs, ltwdiag, lean_gpu, lean_cache and lean_mlp3 and edits none of them (self-test passes). It loads only the cache parts that hold the first questions, without the pool's links, which the probe
does not read.

**Rule, fixed here.** Per dataset, d = recall@5 with the twelve columns minus the baseline's:
- **PAIR_ADDS**: d >= +0.01.
- **PAIR_HURTS**: d <= -0.01.
- **PAIR_NOTHING**: |d| < 0.005. Anything else is MIXED, reported with the numbers. The AUC difference is reported and is not part of the call.
- **What follows.**
  - PAIR_ADDS on musique with PAIR_HURTS on none of the five, or PAIR_ADDS on at least three of the five: the six columns are declared as an input block on the GNN first. That block is GNP, an arm of the
    GNN of record (zg6r) with its own base and null, read by LTG's in-domain rule. The MLP follows second, on the CPU. Each is its own file, since a new input is a model change under the 13 October rule.
  - PAIR_NOTHING on at least three of the five: question-conditional similarity to rrf's top five is closed as an input for models that have the lean columns. The GNN's next form is then LTG's term over
    every pool row, after LTG's verdict.
  - Anything else: reported, and nothing is declared from it.
- **Stated against it.** A logistic probe on all rows is far below the models (it has no interactions), so a gain here can be larger or smaller than a model's. The first questions of each fit carve are
  read once, with no seed. rrf's top five stand in for a ranker's. The musique read is 400 questions.

Items: `pfp-smoke` (hotpotqa, 32 questions), then `pfp-pair-<dataset>` for the five datasets, musique first, one at a time. They are CPU only and write only `outputs/pfs/pair_<dataset>.json`. They
read PKO's cache and the served node vectors (on the host, the verified mirror), read only.

## 12. Results of Amendment A3 (filed 3 November 2026; section 9's fit carves and first questions, every pool row, no seed)

Every read passed the identity: no row was off, and the largest |c.q - dense_cos| was 0.0007 to 0.0010. Every baseline reproduced section 10's recall@5 exactly (gap 0.0000).

| dataset | carve | questions | rows | lean recall@5 | + pair | d | recall@20 | AUC | call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| musique | s1fit | 400 | 2,221,400 | 0.6500 | 0.6663 | +0.0162 | 0.8163 -> 0.8313 | 0.9877 -> 0.9848 | PAIR_ADDS |
| hotpotqa | fit | 1,000 | 711,780 | 0.8053 | 0.8223 | +0.0170 | 0.9229 -> 0.9284 | 0.9871 -> 0.9884 | PAIR_ADDS |
| 2wiki | fit | 1,000 | 1,073,464 | 0.8085 | 0.8203 | +0.0118 | 0.9314 -> 0.9342 | 0.9857 -> 0.9859 | PAIR_ADDS |
| squad | fit | 1,000 | 50,031 | 0.9518 | 0.9558 | +0.0040 | 0.9950 -> 0.9930 | 0.9871 -> 0.9902 | PAIR_NOTHING |
| webqsp | fit | 600 | 2,922,458 | 0.2919 | 0.3300 | +0.0380 | 0.5575 -> 0.5908 | 0.9530 -> 0.9563 | PAIR_ADDS |

- **The call by section 11.** PAIR_ADDS on four of the five (musique, hotpotqa, 2wiki and webqsp) and PAIR_HURTS on none. GNP is declared (docs/GNP_PAIR_COLUMNS_IN_GNN.md), on the GNN first. The MLP
  follows second, on the CPU, in its own file.
- **musique's read is 2-hop only.** Its first 400 questions are all 2-hop (2-hop: 0.6500 -> 0.6663). Its 3-hop and 4-hop questions are not in this read.
- **The single columns beyond rrf's top five** (z-scored AUC; the best two of the six, res_max and pair_mean): musique 0.894 and 0.903, hotpotqa 0.673 and 0.633, squad 0.823 and 0.856, webqsp 0.568
  and 0.551. On 2wiki every column is below 0.5 there (0.37 to 0.48), yet the probe gains: the columns help beside the lean ones, not alone.
- **Stated against it.** Section 11's, unchanged. musique's AUC over all rows fell (0.9877 -> 0.9848) while its recall@5 rose. The reads' JSON stay under `outputs/pfs` (gitignored).

## 13. Amendment A4, 3 November 2026, after section 12 was filed and before any number of the reading it adds

**Why.** Section 12's columns are made against rrf's top five. Two forms already named on them turn on the anchors:
- MPP's next form after NO_GAIN or NOT_ADOPTED (docs/MPP_PAIR_COLUMNS_IN_MLP.md section 7): the same six scores against rrf's top ten rows.
- GNP's next form after ADOPT_FOR_FULL_RUN (docs/GNP_PAIR_COLUMNS_IN_GNN.md section 7): the same six columns against the GNN's own top five after a first pass (two-pass refinement).

Each costs fits: the MLP's two screen fits on the host's CPU, about 2.5 hours and 40 GB each, and the GNN's two on the card. Before either runs, this asks the probe, with no fit and no seed, whether either anchor set carries more than rrf's top five:
- (a) more anchors from the same first stage;
- (b) better anchors: the top five of a first round that already reads the six columns.

(b) is the iterative refinement raised on 2 November, in the probe's terms: rank, re-anchor on the ranking, rank again.

**The read** (`anchors --dataset D`, `anchors_<dataset>.json`; section 9's fit carves and section 11's questions, every pool row). Code: `outputs/mp_unified/zpfq.py`, which imports zpfp, zpfs, ltwdiag, lean_gpu, lean_cache and lean_mlp3 and edits none of them (self-test passes).
- **Pass 1** is section 11's read, unchanged (`zpfp.read`): the 172 lean columns raw and z-scored, A3's twelve columns (P5), the labels and rrf's rank.
- **Three more anchor sets.** Each gives the same six columns, raw and z-scored within the question (twelve), from the same frozen vectors:
  - **P10**: rrf's top ten rows. `pair_1` and `res_1` still read rrf's first row.
  - **R2**: the top five rows by the probe on the lean columns and P5, scored crossfit by question parity (section 9's probe), ties by position. That probe's first row stands for rrf's first in `pair_1` and `res_1`.
  - **R3**, beside, deciding nothing: the same over the top five by the R2 probe's scores.
- **Probes**, section 9's: lean; lean + P5 (section 12's); lean + P10; lean + P5 + R2; lean + P5 + R3; and beside, lean + P5 + P10.
- **Also reported:** the share of rrf's top five kept in each round's top five, and on musique, recall@5 by hop.
- **Two checks refuse the read:**
  - Given rrf's top five, the columns' code must give A3's twelve on the first batch (within 1e-6).
  - lean and lean + P5 must reproduce section 12's recall@5 within 0.002 (A3's own JSON where it is on the machine).

**Rule, fixed here.** Per dataset, against lean + P5's recall@5 in the same read:
- **d10** = recall@5(lean + P10) - recall@5(lean + P5):
  - **TOP10_ADDS**: d10 >= +0.01.
  - **TOP10_HURTS**: d10 <= -0.01.
  - **TOP10_NOTHING**: |d10| < 0.005.
  - Anything else is MIXED.
- **d2** = recall@5(lean + P5 + R2) - recall@5(lean + P5): **R2_ADDS**, **R2_HURTS**, **R2_NOTHING** and MIXED on the same thresholds.
- **What follows from d10.**
  - TOP10_NOTHING or TOP10_HURTS on at least three of the five: MPP's top-ten next form is withdrawn by an MPP amendment before MPP's verdict, and the pair columns' line for the MLP ends at MPP.
  - Otherwise MPP's next form stands as written, with these numbers beside it.
- **What follows from d2.**
  - R2_ADDS on musique with R2_HURTS on none, or R2_ADDS on at least three of the five: R2's columns are declared as an input block on the GNN, in their own file, without waiting for GNP's verdict. Their first round is a fixed probe made on the CPU, as P5's columns are made.
  - For the MLP, a learned first round's top five is a beam chosen by other rows' scores. Under the 2 November ruling only rrf's beam is inside the MLP track, so the MLP's form waits for the user's ruling and is labelled hybrid until then.
  - R2_NOTHING on at least three of the five: re-anchoring on a first round is closed for the pair columns. GNP's two-pass next form is withdrawn by a GNP amendment before GNP's verdict. After ADOPT_FOR_FULL_RUN, GNP's next form is then its message-weight form (its NO_GAIN form).
  - Anything else: reported, and nothing changes.
- R3 and lean + P5 + P10 decide nothing.

**Stated against it.** Section 11's, plus one. R2's anchors for a question come from a probe that did not see its labels. But the R2 probe that scores it was trained on questions whose anchors came from a probe that did see them. That is a second-order leak, through one logistic probe's weights.

**Items.** `pfq-selftest`, `pfq-smoke` (hotpotqa, 32 questions), then `pfq-anchors-<dataset>` for the five datasets, musique first, one at a time, placed before LTA's first screen fit. They are CPU only and write only `outputs/pfs/anchors_<dataset>.json`. They read PKO's cache and the served node vectors (on the host, the verified mirror), read only.

## 14. Amendment A5, 3 November 2026, after section 13 was committed and before any number of the reading it adds

**Why.** Sections 11 and 13 read musique's fit carve on its first 400 questions, and all of them are 2-hop. musique's 3-hop and 4-hop questions are where its gap is widest. In RWT's base reads (zrckpn on WQ-hotpotqa), musique's recall@5 is about 0.58 overall, 0.53 on 3-hop and 0.36 on 4-hop. Section 13 (b) asks a chain question: a later hop needs the hop before it as its anchor. A 2-hop read can understate it.

**The read** (`h34`, `anchors_musique_h34.json`). Code: `outputs/mp_unified/zpfr.py`, which imports zpfq, zpfp, zpfs, ltwdiag, lean_gpu, lean_cache and lean_mlp3 and edits none of them (self-test passes).
- **The questions.** musique/s1fit's cache parts 20, 25 and 26, loaded alone: 242 3-hop and 271 4-hop questions, about 3.1M pool rows. The read refuses parts that hold other counts.
  - The carve's 4,601 questions are ordered by hop.
  - 2-hop: parts 0 to 18 and the first 82 of part 19.
  - 3-hop: the rest of part 19 through the first 70 of part 25.
  - 4-hop: the rest.
- **How the parts load.** Inside the carve's load, lean_cache.part_dirs returns the three parts with their records' row ranges moved to start at 0. The arrays and their shas are unchanged, and it is put back after.
- **The read itself** is zpfq's, unchanged: pass 1, P10, R2, R3, the same six probes, and d10 and d2 against lean + P5. Added: section 11's PAIR call on d_pair = recall@5(lean + P5) - recall@5(lean). Each is also given by hop.
- **Checks.** Only zpfq's machinery check refuses. There is no A3 read of these questions to reproduce.

**Rule, fixed here.** The pooled read (513 questions) gives a PAIR call, a TOP10 call and an R2 call, on section 11's and section 13's thresholds. They join section 13's musique read:
- musique counts as R2_ADDS (or TOP10_ADDS) in section 13's "what follows" when either musique read gives that call and neither gives the HURTS call.
- It counts as R2_NOTHING (or TOP10_NOTHING) only when both reads do.
- The PAIR call is reported beside GNP's and MPP's verdicts. Their own screens read musique's 3-hop and 4-hop slices and decide.
- The by-hop calls decide nothing (242 and 271 questions).

**Stated against it.** Section 13's, plus two:
- These parts are not a random sample of musique's 3-hop and 4-hop questions. They follow the carve's order, and the carve is a fixed subset of musique's train split.
- The probe is crossfit by question parity within these 513 questions only.

**Item.** `pfq-anchors-musique-h34`, CPU only, after `pfq-anchors-musique` and before LTA's first screen fit. It writes only `outputs/pfs/anchors_musique_h34.json`.
