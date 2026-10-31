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

(to be filed after the reads)

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
