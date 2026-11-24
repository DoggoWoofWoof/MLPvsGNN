# SEM2 (sketch): edges as learned operators on any frozen vectors

**Status: a sketch, written 18 November 2026, 20:16 IST. R1 was declared on 23 November (section 7) before any number; R2 and R3 are not declared yet.** The rules,
reads and calls are fixed on this page before any fit runs. This is not the journal's work: the user sees it as a
possible separate paper. Nothing here enters D83's pools.

## 1. The user's direction (18 November)

- "we refine the vectors according to the links, so the neighbour search would be a semantic space traversal kinda like
  how attention works and how it learns relations but i want it learn edges, the novelty would be to not finetune a
  encoder but start from any set of vectors"
- "smoothing the original doc vector with neighbours i do not think it would work, because we are limited by the
  dimentions, so if we add extra dims then it would make sense but attention type of softmax scoring so that we know
  the links to actually focus on"
- "the edges is what i am thinking can be approximated by a function or mathematical rotation etc how it works in
  transformers or anything better if exists"

SEM1 (filed the same evening, [SEM1_GRAPH_SPACE.md](SEM1_GRAPH_SPACE.md) section 12) agrees with the second point. A
fixed smoothing in the same dimensions gives no top-5 gain on any arm. It only reaches a little deeper.

## 2. What is already on file

| stage | form | outcome | what it means for SEM2 |
| --- | --- | --- | --- |
| P5a | free per-node vectors and translation offsets (TransE) in a 64-wide PCA frame; the offsets chosen from the question alone | STOPS: the chooser collapsed to one chain per corpus. The oracle chain beat the hops on 5 of 6 datasets | the space held relations; the chooser was blind to the seed |
| P5b, P5b2, P5b3 | a learned offset g(question, seed, graph), as a reranker | more of the low golds on MetaQA, 2Wiki and HotpotQA, fewer on MuSiQue. The fit memorised seen golds | stop on held-out new golds |
| P5c, P5c5 | g's columns as an input to both models | NOT_ADOPTED, no GAIN | the rerankers already read hops and cosine |
| SEM1 | a fixed mean over the links, same dimensions, first stage | closes: no top-5 gain; u's R@1000 a little above v's | a fixed mix cannot choose a link |

**Lessons carried in:**
1. Act in the first stage. That is where the small encoders lose (ENC, section 8), and the rerankers cannot reach it.
2. Keep the full vector and add dimensions. Never squeeze into a 64-wide frame.
3. No free per-node parameters: they cannot transfer to a new corpus or encoder.
4. The chooser sees the question and the seed.
5. Stop the fit on held-out new golds.

## 3. The form (a sketch)

**Edges as operators.**
- In a transformer, a head scores a pair as (W_Q x_i) · (W_K x_j), a learned bilinear form, and single heads come to
  track single relations.
- RoPE rotates queries and keys, so a score depends on the pair's relative position.
- Knowledge-graph embeddings give each relation an operator: a translation (TransE), a rotation (RotatE), a bilinear
  form (DistMult, ComplEx) or reflections (HousE).

SEM2 gives each edge label one operator on the frozen vectors:

    score_r(s, d) = (T_r P v_s) · (P v_d)

- **P** is a shared map from the encoder's D dimensions to D' ≥ D. These are the extra dimensions. At the start, P =
  [I; 0], so the first stage starts as today's.
- **T_r** is a rotation in D': block-diagonal 2×2 planes, so D'/2 angles per label.
  - A rotation keeps length, so the cosine search stays valid.
  - Rotations compose: T_r2 T_r1 is a two-hop operator.
  - Translation (P5a's family) and a low-rank bilinear form are read beside it, so the choice of rotation is tested,
    not assumed.
- **Labels** are D83's rule labels: S.<rel> where the data has structure, R.* everywhere, never K, which the encoder
  makes. For a label never seen in training (a new KB's relations), the angles come from the label name's frozen
  vector through a small shared net. A new relation then gets an operator without a fit.
- **The operators fit from the links alone, at index time, per corpus.**
  - Positives: the corpus's edges with label r.
  - Negatives: random nodes and nodes near in the dense space.
  - No question and no gold is read. Any corpus with links, and any vector set, gets its operators.

**Attention: which link to follow.** For question q and seed s (the first stage's top seeds):
- α(q, s) is a softmax over the labels, from a small net on: q against each moved seed T_r P v_s, the label name's
  vector, and the seed's degree per label.
- The traversal query is t(q, s) = Σ_r α_r T_r P v_s + λ P q.
- One nearest-neighbour search per seed runs over the index of P v_d.

That search is the "semantic space traversal". The question picks the relation, the operator moves the seed, and the
search finds the documents at the end of that edge. Those include documents the rule links never joined.

**What learns from questions:** only α (a few thousand weights, shared across corpora and encoders) and λ.
- They train on the training carves' golds.
- They are cross-fitted leave-one-dataset-out.
- The fit stops on held-out new golds (P5b2's stop).

## 4. The reads (fixed at declaration)

- **R1, link recovery (no QA).** For held-out edges of each label: the rank of the true tail among all documents
  under the operator, against the frozen cosine. Read per dataset and per arm. This answers whether operators on frozen
  vectors learn the edges, for text and for KB graphs.
- **R2, the first stage.** As SEM1:
  - base = RRF(v, SPLADE), set against RRF(v, trav, SPLADE) and a control;
  - on MuSiQue, SQuAD and MetaQA;
  - also on 2Wiki and HotpotQA where vectors exist (frozen now, mpnet once its encodes run).
- **R3, transfer:**
  - (a) the left-out dataset, with α fit without it;
  - (b) another encoder's vectors, with the operators refit from the links (no QA) and α unchanged;
  - (c) unseen relations through the name net, holding out a share of MetaQA's and WebQSP's relations.
- **Beside:** SEM1's add (filed), kNN database-side augmentation, and the translation and bilinear operators.
- **Cost:** CPU, small.

## 5. Prior work and what may be new (checked 18 November; recheck before any claim)

**Known:**
- **Operators per relation:**
  - TransE (Bordes et al. 2013), RotatE (Sun et al., ICLR 2019), ComplEx (Trouillon et al. 2016) and HousE (ICML
    2022);
  - rotations inside attention: RoPE (Su et al., RoFormer 2021);
  - attention heads that track relations (Clark et al. 2019).
- **Rotation angles from a relation's text vector:** EMNLP 2026 main 437, for temporal KGs.
- **Inductive and text-based link prediction:**
  - BLP (Daza, Cochez and Groth, 2021, arXiv 2010.03496), which trains the text encoder;
  - ULTRA (Galkin et al., ICLR 2024, arXiv 2310.04562): relation-relative representations transfer zero-shot across 57
    KGs;
  - multi-hop queries answered by composing a link predictor: CQD (Arakelyan et al., ICLR 2021; not rechecked
    today).
- **Learned links over frozen passage vectors for multi-hop: Dury 2026 (arXiv 2604.20850).** This is the closest work,
  and the warning.
  - An MLP over frozen embeddings, trained contrastively on co-occurrence, used as a reranker.
  - HotpotQA R@5 goes from 0.831 to 0.916, and MuSiQue gains +10.1 transductively.
  - There is no significant gain inductively: it memorised the corpus.
- **Training-free neighbour residuals over frozen embeddings:** NEAR (arXiv 2610.13314), for memory retrieval, over
  nine embedding models.
- **Graph trained into the encoder:** HLP, LinkBERT, SPECTER and SciNCL (SEM1, section 11).

**Not found (a possible contribution):**
- typed operators learned from rule-built links, with no LLM and no QA labels, over any frozen passage vectors, used as
  a first-stage traversal search for multi-hop QA;
- the same form across text corpora and KBs and across encoders, with transfer read on unseen datasets, encoders and
  relations;
- small-encoder gap recovery through the index rather than the encoder.

## 6. Open before the declaration

- D' (how many extra dimensions) and the operator families read.
- Seeds per question (P5b's 5 dense and 5 SPLADE) and hops (one, then two composed).
- Labels on MuSiQue and SQuAD: R.* only (the 10 November ruling: no S there).
- R1's negatives and its held-out share of edges.

## 7. R1 declared: do operators on frozen vectors learn the edges? (23 November 2026, 17:09 IST, 11:39 UTC)

The user started SEM2 on 23 November ("you can start sem2"). D101's encoder is fixed (e3v), and another account holds
the GPU. This section declares R1, the link-recovery read of section 4, before any SEM2 number exists. It closes
section 6's open points for R1. R2 (the first stage with the attention α) and R3 (transfer) are declared later, after
R1 is filed. Nothing here enters the journal's pools or the one refit.

**What had been seen when it was written:**
- SEM1's filed reads (section 2);
- the link records of UL1, UL2 and MHR1, counts only;
- the first-stage reads of E4 and E6.

No operator has been fit. No edge has been ranked under any scorer.

### 7.1 Vectors (two arms)

- **e3v** is the encoder (D101): D = 1,152. It joins gte-modernbert-base (768) and
  granite-embedding-small-english-r2 (384), each half scaled by sqrt(1/2), as stored.
- **frozen** (gte-Qwen2-1.5B, D = 1,536) is read beside it, as everywhere since D101.
- Rows stay float16 in memory, as stored. Each row is cast to float32 and divided by its norm when read. A row whose
  norm is off 1 by more than 1e-2 stops the dataset.

### 7.2 Edges and labels (the records' own pairs, never K)

**Sources:**
- The six: UL1's checked families (zul1.families, which checks every file's sha256).
- MultiHop-RAG: MHR1's rule files, checked against links.json's sha256s.

**Labels:**
- **S.<rel>:** the structural store's pairs with their relation, directed src -> dst as stored. Read only on the four
  datasets A1 gives S: 2Wiki, HotpotQA, MetaQA and WebQSP. MuSiQue's and SQuAD's stored title mentions have not been S
  since A1, and are not read (section 6).
- **R.mention:** W2_S. **R.name_comention:** CN_all. **R.title_comention:** F0T minus CN_all.
  - All three are undirected pair sets in the records, so each is read in both directions.
  - On MultiHop-RAG: F0 is CN_all, and F0T minus F0 (75 pairs) is the title label. Its W2_S has 2 pairs.
- **Not read:** K.knn (the encoder makes it) and R.hub (not carried; it joins R1 only by a later section).

**Operators:**
- A label with at least 500 training edges gets its own operator.
- S relations with fewer are pooled into one operator, S.rare.
- R labels with fewer are counted, not read (so MultiHop-RAG reads R.name_comention only).

**Read labels (at most 16 per dataset):**
- every R label with an operator;
- the 12 S operator labels with the most training edges;
- S.other: all remaining S edges, each scored by its own operator (or S.rare's).

### 7.3 The split: node-inductive (section 2's lesson; Dury 2026's warning, section 5)

- **Node split:** each node takes u = the first 8 bytes of sha256("sem2|<dataset>|<node index>"), read as an integer
  over 2^64. It is **test** if u < 0.10, **dev** if u < 0.20, and **train** otherwise.
- **Edges:**
  - **train:** both ends are train nodes;
  - **dev:** the source is a dev node and the target is not a test node;
  - **test:** the source is a test node.
  - An undirected pair gives an edge from each end that qualifies.
- No test node is an end of any training or dev edge, and training negatives are train nodes only. So no score can
  come from a memorised node.
- **Samples:** up to 1,000 test edges and 500 dev edges per read label (numpy default_rng(0), without replacement).

### 7.4 The model

**The shared map:**
- x = P v / ||P v||, with P in R^(D' x D) and **D' = D + 256** (the extra dimensions).
- P starts as [I; 0], so x starts as v.

**Rotation (the form read):**
- T_r is block-diagonal, with a 2 x 2 rotation on coordinates (2i, 2i+1) by the angle theta_r,i. Each label has D'/2
  angles, all starting at 0.
- **Directed labels (S):** score_r(s, d) = (T_r x_s) . x_d.
- **Undirected labels (R):** the score is the symmetric part, 1/2 [(T_r x_s) . x_d + (T_r x_d) . x_s]. That equals
  the sum over planes of cos(theta_r,i) times the plane's dot product. On R labels, the rotation is therefore a
  per-plane weighting.
- **At the start every score is exactly the cosine v_s . v_d** (gate G-0). The fit moves away from today's space
  only where the links ask for it.

**Beside (separate fits, the same rules):**
- **translation:** score = (x_s + b_r) . x_d. On undirected labels it is the symmetric part, so the query is
  x_s + b_r / 2. b_r starts at 0, and P is the same shared map.
- **no extra dimensions:** rotation with D' = D (the user's 18 November point, "limited by the dimentions").
- **map only:** P is fit with every theta fixed at 0. This asks whether the gain is the shared map or the per-label
  operators.
- The last two run on the four small sets only: MultiHop-RAG, SQuAD, MuSiQue and MetaQA.

No free per-node parameter appears anywhere (section 2's lesson 3).

### 7.5 The fit (links only; no question, no gold)

- **Loss:** softmax cross-entropy at temperature 0.05.
  - Each positive (s, r, d) competes with the other in-batch tails (1,024 positives a batch, labels mixed) and with
    4,096 random train nodes shared by the batch.
  - A candidate that is a true r-tail of s is masked, checked against the label's pairs.
  - An undirected training pair is oriented at random (seeded).
- **Optimiser:** Adam, learning rate 1e-3, float32, 8 threads, seed 0.
- **An epoch:** up to 200,000 training edges, an equal share per read label. A label with fewer gives all it has.
  Each share is drawn without replacement, with seed = the epoch.
- **Stop:** after each epoch, the read of section 7.6 runs on the dev sample (dev MRR). The fit stops after two
  epochs without a gain, or at most 10, and keeps the best epoch.

### 7.6 The read

- For each sampled test edge (s, r, d): the rank of d among all n nodes by score_r(s, .).
  - Filtered: s itself, and s's other true r-tails in any split, are removed.
  - Ties count half.
  - Measures: MRR, hits@10 and hits@100.
- **Scorers:**
  - cosine, v_s . v_d (today's space, the baseline);
  - the rotation;
  - the beside fits.
- Results are given per read label, and pooled over the read labels' samples.

### 7.7 The call (per dataset and arm)

- **Delta = MRR(rotation) - MRR(cosine) over the pooled test sample.**
  - Paired bootstrap over test edges: 2,000 resamples, seed 0, 95% interval.
  - ABOVE = **LEARNS**; LEVEL = **LEVEL**; BELOW = **LOSES**.
  - Beside: the per-label deltas, hits@10 and hits@100.
- **R1 per arm:** HOLDS if it LEARNS on all seven, PARTIAL if it LEARNS on at least one, FAILS if it LEARNS on none.
- **Expectations, written before any number.** If one fails, the failure is reported, not explained away.
  - X1: e3v LEARNS on MetaQA and WebQSP. Typed, directed relations are a rotation's home ground.
  - X2: e3v LEARNS on MuSiQue.
  - X3: on the small sets, rotation with D' = D + 256 is at or above rotation with D' = D.
  - X4: frozen's delta is below e3v's on most datasets. A larger encoder leaves less to learn.

### 7.8 Gates (a failure stops that dataset and arm)

- **G-E:** each family's pair count equals the record's.
  - The six: UL1's grade.json family_pairs.
  - The directed S read gives exactly zul1.s_labels' (pair, relation) rows.
  - MultiHop-RAG: links.json's F0, F0T and W2_S counts and sha256s.
- **G-S:** no test node is an end of any training or dev edge, and no test or dev node is a training negative.
- **G-0:** before the fit, each operator scorer must match cosine on the dev sample. The true-tail scores agree within
  1e-4, and the MRRs within 0.001.
- **Self-test:** a toy with planted operators: one directed label (a rotation) and one undirected label (plane
  weights).
  - The fit must recover them: rotation MRR far above cosine on the toy's test edges.
  - G-0 and G-S must pass on the toy.

### 7.9 What follows (fixed now)

- **HOLDS or PARTIAL on e3v:** R2 is declared before any R2 number. It is the traversal of section 3, with the
  operator form R1 supports, on all seven datasets (the universal rule).
- **FAILS on e3v:** the rotation form closes. If a beside fit LEARNS where the rotation does not, it is named as the
  next form and declared separately.
- No call here touches the journal's layout or pools.

### 7.10 Code, rows and outputs

- **Code:** outputs/mp_unified/zsem2.py (new). It imports zul1, zul2, zmhr and zenc and edits none of them. Commands:
  `selftest`, `r1 --ds D --arm A`, and `report`.
- **Rows:** host CPU, 8 threads. The e3v vectors of the six are on the host only.
  - sem2-r1-<ds>-<arm>, the small sets first: MultiHop-RAG, SQuAD, MuSiQue and MetaQA, then WebQSP, HotpotQA and
    2Wiki. Then sem2-r1-report.
  - Estimates: under 1.5 h per small-set row with its beside fits, and 2-4 h per large-set row.
  - Memory: the largest rows are 2Wiki's frozen vectors (18.4 GB in float16) and WebQSP's W2_S (500 million pairs).
- **Outputs:** outputs/sem2/r1/<ds>/<arm>.json, report.json and report.md. The json and md files are committed; the
  arrays (npz) are not.

### 7.11 Amendments before any R1 number (23 November 2026, 17:21 IST, 11:51 UTC)

zsem2.py's self-test found two places where the form of 7.4 could never move from the cosine. Both start points had
zero gradient, so the fit would have read the cosine and called it LEVEL by construction. They are amended here, before
any row runs.

- **(a) The extra dimensions.** If P's 256 extra rows start at exactly 0, every node's extra coordinates are 0, and the
  gradient of every score with respect to those rows is 0 in every form. They never move. They now start at
  N(0, 1e-4^2) per entry (seed 0). The start still equals the cosine within G-0's tolerance: the self-test's largest
  score change is 1.5e-7.
- **(b) The undirected operator.** The symmetric part of a rotation weights plane i by cos(theta_r,i). At
  theta = 0 its gradient, -sin(theta), is 0, so it never moves either. On undirected (R) labels the operator is now a
  free weight per plane, w_r,i, starting at 1. Each label has D'/2 weights, the same count as the angles. At the start
  it is the cosine. The weights are unbounded, so a label may also sharpen or flatten its scores; that does not change
  any rank within the label. Directed (S) labels keep the rotation of 7.4.

**The self-test after (a) and (b).** A toy of 4,000 nodes in 32 dimensions plants a rotation on one directed label and
plane weights on one undirected label, with a margin (each tail is the moved source plus noise).
- The cosine reads MRR 0.27, the rotation 97.24: +96.98 [+95.01, +98.64], LEARNS.
- Translation reads 36.59 and the map alone 34.45.
- Every rank the read computed equals a full sort's.
- G-0 and G-S pass.

**A smoke run, not a read.** One run on MetaQA's frozen vectors checked the real shapes on the laptop: the loaders,
G-E, the split, the read groups, the left-out tails, the fit loop and the json. It used the smoke settings (one epoch
of 4,024 edges, 60 test and 40 dev edges per label) and wrote only to a scratch folder. Its numbers are not R1's and
are not read.

- **What it showed about the shapes:**
  - MetaQA reads 10 groups: R.mention, R.name_comention, 7 S relations, and S.other.
  - Its F0T equals CN_all, so it has no R.title_comention.

## 8. R1h declared: hubs (24 November 2026, 14:43 IST, 09:13 UTC)

**The user (24 November), on WebQSP's R1 rows:** "we need to mitigate that and somehow normalize it, it isnt good for
retrieval if we always land in the hub, it is a start which means it identifies the right space now we have to
identify the right vectors, need to downweight hub".

**What had been seen when it was written:**
- R1's json records for 11 of the 14 rows: MultiHop-RAG, SQuAD, MuSiQue, MetaQA and WebQSP on both arms, and HotpotQA
  on e3v. Every one LEARNS on 7.7's pooled MRR.
- Their per-label deltas. On WebQSP the largest gains are on relations whose edges point at a few targets: person
  gender +78.54 (e3v), film starring +87.66, topic notable for +64.81. The text links gain less: R.mention +14.02.

No number grouped by target has been computed, and no hub-aware fit has run.

**Why this matters.** 7.6's pooled MRR counts every test edge once. A label whose edges point at a few hub targets
(person gender has two) is learned by aiming the operator at the hub. That helps no retrieval: a traversal that lands
on "male" has found nothing. R2 needs the operator to find the right node among many, not the hub.

### 8.1 Definitions

- **deg_L(y):** the number of label-L training edges (7.3: both ends train nodes) whose target is y.
  - An undirected R label counts y's training pairs.
  - An S relation counts training edges with destination y.
  - Only training edges count, so no test or dev edge informs it.
- **Target-macro MRR (MRR_T):**
  - The test sample's edges are grouped by (read group, target). A group's value is the mean reciprocal rank of its
    edges.
  - MRR_T is the mean over groups. A hub counts once per read group, however many sampled edges point at it.
- **Its interval:** the difference of two scorers' group values, paired bootstrap over groups (2,000 resamples, seed
  0, 95%). ABOVE, LEVEL and BELOW as in 7.7.
- **Pooled MRR** (7.6) is reported beside every MRR_T and is never called on here.

### 8.2 Part A: the hub read of R1 (no fit)

From R1's saved ranks (outputs/sem2/r1/<ds>/<arm>.npz: test_grp, test_d and rank_<form>):
- MRR_T of the cosine and of every R1 form.
- **Call H0:** R1's rotation against the cosine, on MRR_T: LEARNS, LEVEL or LOSES.
- **Descriptive, per read group:** the sampled edges, their distinct targets, and the share of edges whose target
  takes 10 or more of the group's sampled edges.

### 8.3 Part B: downweighting hubs at query time (no new fit)

Logit adjustment (Menon et al., 2021), with no tuned value:
- R1's logits are q . x_y / tau, trained against negatives drawn uniformly from the training nodes. So a label's hub
  prior, log deg_L(y), sits inside the learned score.
- **The adjusted score:** q . x_y - tau log(1 + deg_L(y)), with tau = 0.05, R1's temperature (7.5). The 1 keeps a node
  with no training edge of the label at no penalty.
- The cosine gets the same correction, beside.
- The ranks follow 7.6: filtered, ties half.
- CSLS (Conneau et al., 2018) was considered and is not run. It needs every node's similarity to thousands of moved
  sources per operator, and WebQSP has 791 operators: about 10^15 multiply-adds per read.

### 8.4 Part C: the hub-balanced fit (HB)

R1's rotation (7.4, 7.5), with two changes:
- **The loss:** each training positive (s, L, d) is weighted w = 1 / deg_L(d), normalised within the batch:
  sum(w CE) / sum(w). Edges are still sampled uniformly, so in expectation each target carries the same weight,
  whatever its degree.
- **The stop:** dev MRR_T instead of dev MRR.

Everything else is unchanged: the split, the samples, the negatives, the temperature, the epochs, D' = D + 256 and the
seed. HB is read raw and with the adjustment (8.3).

**Beside: popularity.** Every node ranked by deg_L alone (ties half, filtered as 7.6). This is what a scorer that
knows only the hubs reads.

### 8.5 Gates

- **G-R (reproduction):**
  - zsem2h.py rebuilds R1's split, plan and samples from the same graph. They must equal R1's saved test_s, test_d and
    test_lab exactly.
  - R1's rotation, reloaded from its npz, must reproduce R1's saved test ranks: at least 99.9% identical, and the
    pooled MRR within 0.01 points.
- **G-0, G-S and G-E** as in 7.8, for the HB fit.
- **G-C:** on the self-test, the adjusted ranks and the popularity ranks equal a full sort.

### 8.6 The calls (per dataset; e3v decides, frozen is read beside)

- **H0:** R1's rotation against the cosine, on MRR_T (part A).
- **H1:** HB against R1's rotation, both raw, on MRR_T.
- **H2:** R1's rotation with the adjustment against R1's rotation raw, on MRR_T.
- **Beside:** HB with the adjustment against R1 raw; the cosine with the adjustment against the cosine; popularity;
  and every one of these on pooled MRR.
- **Carry rule (universal).** HB and the adjustment are judged separately. Each is carried into R2 if, on e3v, it reads ABOVE on
  at least 4 of the 7 datasets and BELOW on none. Otherwise it is carried nowhere. If both are carried, R2 uses HB
  with the adjustment.
- **Expectations, written before any number:**
  - X5: H0 is LEARNS on at least 4 of the 7 datasets (e3v), and WebQSP's MRR_T gain (e3v) is below half its pooled
    gain (+39.07).
  - X6: H2 is ABOVE on WebQSP (e3v).

### 8.7 What does not change

- R1's own call (7.7) stays on pooled MRR, as declared. Its filing reports MRR_T beside it, with this section's caveat.
- Nothing here enters the journal's pools or the one refit.

### 8.8 Code, rows and outputs

- **Code:** outputs/mp_unified/zsem2h.py (new). It imports zsem2 and edits nothing. Commands: `selftest`, `hubread`,
  `r1h --ds D --arm A` and `report`.
- **Rows:**
  - sem2h-selftest;
  - sem2h-hubread, over the R1 npz files present when it runs (the report reads all 14);
  - sem2h-r1h-<ds>-<arm>, each after its R1 row, e3v first and then frozen, the small sets first;
  - sem2h-report.
- **Memory** as R1's rows.
- **Outputs:** outputs/sem2/r1h/. The json and md files are committed; the npz files are not.

### 8.9 Amendment: the carry rule (24 November 2026, 15:38 IST, 10:08 UTC), before any H1 or H2 number was read

**The user, after part A:** "yeah so we have to find something that works with all datasets including metaqa".

**What had been seen when it was written:**
- Part A (8e64299) only:
  - MetaQA's gain halves on MRR_T: +12.31 pooled, +5.63 MRR_T (e3v).
  - WebQSP's barely moves: +39.07 pooled, +35.43 MRR_T.
  - On the other four datasets read, the two agree within 0.5 points.
- The sem2h-r1h rows had started (MultiHop-RAG e3v at 14:58). None of their output had been read.

**Why the rule changes.** A hub fix can only help where hubs exist. On datasets where MRR_T and pooled MRR agree, LEVEL
is the best it can do. So 8.6's "ABOVE on at least 4 of the 7" asks for gains that cannot exist.

**The carry rule (replaces 8.6's).** HB and the adjustment are judged separately, on e3v, on MRR_T. Each is carried
into R2 if:
- it is ABOVE on MetaQA; and
- it is BELOW on none of the seven datasets.

Otherwise it is carried nowhere. WebQSP, the frozen arm and pooled MRR are read beside; they do not decide.

**If both are carried:**
- R2 uses HB with the adjustment (hb_la), provided hb_la against R1 raw is also ABOVE on MetaQA and BELOW on none.
- If it is not, R2 uses whichever of the two has the larger MetaQA gain.

**If neither is carried:**
- No R2 is declared on hubs.
- The next hub step is declared from the per-label reads (which labels move, which lose) before any new fit.

**Order and code:**
- The e3v rows of MetaQA and WebQSP run first, then the other e3v rows, then the frozen rows.
- outputs/mp_unified/zsem2hr.py (new) writes outputs/sem2/r1h/report.json and report.md under this rule. It shows
  8.6's rule beside it, marked superseded.
- zsem2h.py has run rows and is not edited, and its own `report` command is not run. The sem2h-report row runs
  zsem2hr.py.

### 8.10 The calibrated adjustment (R1c; 24 November 2026, 15:47 IST, 10:17 UTC), before any R1c number

**What had been read when it was written:**
- Part A (8e64299).
- R1h on two datasets, e3v, MRR_T:
  - MultiHop-RAG: H1 -0.24 LEVEL; H2 -7.28 BELOW.
  - SQuAD: H1 -3.78 BELOW (R.mention -6.96); H2 -10.88 BELOW; HB with the adjustment -13.62 BELOW.
- So under 8.9, HB, the adjustment and the two together are all NOT_CARRIED, whatever the other rows read. No other
  R1h output had been read (MuSiQue e3v was at HB epoch 1).

**Why.** On the co-mention labels (R.*), a node's degree goes with being a true target. The rotation holds no popularity
prior there to remove, so the full penalty takes signal away. On MetaQA's hub relations, the same penalty is what part A
asks for. One fixed strength cannot serve both. So the strength is chosen per read group, on dev, and zero is always one
of the choices.

**The method (no fit):**
- R1's saved rotation and the cosine, each as it is.
- For each read group g, a strength beta_g in {0, 0.25, 0.5, 0.75, 1}. The score is
  q . x_y - beta_g tau log(1 + deg_L(y)), with tau = 0.05 and deg_L as in 8.3.
- beta_g is the one with the largest MRR_T on R1's dev sample. A larger beta must beat every smaller one strictly, so
  ties go to the smaller. beta is never negative: hubs are never pushed up.
- The test read uses each group's beta_g. beta = 0 everywhere is R1 itself.
- The cosine gets its own betas, chosen the same way (beside).

**The call (per dataset; e3v decides, frozen is read beside):**
- **H3:** the calibrated rotation against R1's rotation raw, on MRR_T, with 8.2's paired bootstrap.
- **Beside:** the calibrated cosine against the cosine; the calibrated rotation against the cosine; the rotation against
  the cosine; and every one of these on pooled MRR.

**Carry (as 8.9).** The calibrated adjustment is carried into R2 if, on e3v and MRR_T, H3 is ABOVE on MetaQA and BELOW
on none of the seven datasets.
- If it is carried, R2 uses it, with the betas chosen again on R2's own dev sample (same grid, same tie rule).
- If it is not, R2 carries no hub step, and R1's filing reports MRR_T beside pooled MRR. Any further hub step is
  declared from R1c's per-group reads before any fit.

**Gates:**
- **G-R** as in 8.5. The dev sample must also be rebuilt from the same graph and seed.
- **G-B:** at beta = 0, the one-pass ranks must equal zsem2's own rank read exactly, on dev and on test, for the
  rotation. For the cosine: exactly on dev, and at least 99.9% identical to R1's saved cosine ranks on test.
- **G-C:** on the self-test, ranks with a different beta per query must equal a full sort, for the cosine and for a
  trained rotation.

**Expectations, written before any number:**
- X7: beta = 0 on every R group of SQuAD, MuSiQue and MultiHop-RAG (e3v, rotation).
- X8: H3 is ABOVE on MetaQA (e3v).

**R1h rows dropped (they cannot change 8.9's verdict):**
- Not yet run: WebQSP, HotpotQA and 2Wiki e3v, and all seven frozen rows.
- MuSiQue e3v was running (HB at epoch 1). It is cancelled.
- MetaQA e3v still runs. Its H1 tells whether a fitted hub fix helps where the hubs are, for the step after this one.
  It decides nothing here.
- sem2h-report runs over the rows present. SQuAD's BELOW already sets its verdict.

**Code, rows and outputs:**
- **Code:** outputs/mp_unified/zsem2hc.py (new). It imports zsem2h and zsem2 and edits neither. Commands: `selftest`,
  `r1c --ds D --arm A` and `report`. The self-test passed on the laptop before this was committed: G-R, G-B and G-C on
  the toy.
- **Rows:**
  - sem2hc-selftest;
  - sem2hc-r1c-multihoprag-e3v first: it is small and checks the real path;
  - then the other e3v rows, MetaQA and WebQSP first, each after its R1 row;
  - then the frozen rows;
  - sem2hc-report after the seven e3v rows, and sem2hc-report-all after all fourteen.
- **Memory** as R1's rows.
- **Outputs:** outputs/sem2/r1c/. The json and md files are committed; the npz files are not.

### 8.11 The hub ceiling (R1k; 24 November 2026, 16:21 IST, 10:51 UTC), before any R1k number

**What had been read when it was written:**
- R1h, filed as 706425c. HB, the adjustment and the two together are all NOT_CARRIED. On MetaQA all three are BELOW,
  and they lose most on its hub relations (has_genre: HB -23.44, the adjustment -22.58).
- R1c on MetaQA, MultiHop-RAG, SQuAD and MuSiQue, both arms:
  - Dev picks beta = 0 on every group but one (MetaQA e3v's S.written_by, 0.25).
  - H3 is LEVEL on seven rows and BELOW on MetaQA e3v (-0.02). So 8.10's adjustment is NOT_CARRIED.
  - X7 holds; X8 fails.

**Why this read.** 8.9 asks for the next hub step to be declared from the per-label reads. Those reads show every
degree penalty losing, even on the relations whose answers are hubs. Two explanations remain:
- (a) Hubs do crowd out the right answer. But a penalty shared by every query of a group cannot remove them without
  also pushing down the queries whose answer is a hub.
- (b) Hubs seldom sit above a less popular right answer. Then MetaQA's gap between pooled MRR and MRR_T only means that
  popular answers are easier to learn.

The ceiling tells the two apart without a fit.

**The ceiling (no fit, no parameter).**
- Take any adjusted score s(y) - f(deg_L(y)) with f non-decreasing; 8.3's and 8.10's are of this form.
- A node no more popular than the true target cannot move from above the target to below it.
- So, per test edge, the rank counted only over the nodes with deg_L <= deg_L(target) is the best rank any such
  adjustment can reach, even one chosen separately for each query.
- **C** = MRR_T(ceiling) - MRR_T(R1's rotation), with 8.2's paired bootstrap.

**The reading (e3v decides; the frozen arm is not run):**
- **ROOM** on a dataset if C >= 1.0 MRR_T point. This uses the point estimate, which favours hubs, since C is already
  an upper bound.
- **If no dataset has ROOM,** hub work is closed. No degree-based score step can gain a point on any of the seven, and
  R2 works on the operators themselves.
- **If some do,** the next hub step is declared before any fit, on the labels of those datasets with the largest
  per-group C. One example would be a gate that decides, per query, whether the answer is a hub.
- **Beside:**
  - the cosine's ceiling;
  - C on pooled MRR;
  - the share of test edges with any more popular node above the target ("crowded");
  - the share of targets with degree 0.

**Expectation, written before any number:** X9: no ROOM on SQuAD, MuSiQue or MultiHop-RAG.

**Gates:**
- **G-R** as in 8.5, on the test sample.
- **G-B:**
  - The one-pass raw ranks must equal R1's rotation ranks exactly. For the cosine, they must be at least 99.9%
    identical to R1's saved cosine ranks.
  - The ceiling is never worse than raw.
- **G-C:** on the self-test, the raw and ceiling ranks equal a full sort, and the ceiling moves at least one rank.

**Code, rows and outputs:**
- **Code:** outputs/mp_unified/zsem2hk.py (new). It imports zsem2h and zsem2 and edits neither. Commands: `selftest`,
  `r1k --ds D --arm A` and `report`. Its self-test passed on the laptop before this was committed.
- **Rows:**
  - sem2hk-selftest;
  - sem2hk-r1k-<ds>-e3v for all seven datasets. MultiHop-RAG runs first as the real-path check, then MetaQA, SQuAD
    and MuSiQue, then WebQSP, HotpotQA and 2Wiki;
  - sem2hk-report after the seven.
  - They are queued ahead of the R1c rows still waiting. Those still run as declared, and their report follows them.
- **Memory** as R1's rows.
- **Outputs:** outputs/sem2/r1k/. The json and md files are committed; the npz files are not.

### 8.12 Post hoc: the ceiling over seen and unseen targets (24 November 2026, 16:56 IST, 11:26 UTC)

This section is written after reading 8.11's first four rows. It decides nothing, and 8.11's verdict stands as read.

**What had been read (R1k, e3v; C in MRR_T points):**
- MultiHop-RAG +4.93, ROOM. SQuAD +4.86, ROOM. MuSiQue +2.16, ROOM. So X9 fails.
- MetaQA +0.90, NO_ROOM. Its hub relations are far above that: has_genre +11.72, in_language +7.46,
  release_year +6.99.
- On the co-mention datasets, the room is on labels with no hubs in the test sample (hub share 0.0).

**Why a split.** 7.3's test edges start at a test node and may end at any node. So about a fifth of the targets are test
or dev nodes, whose training degree is 0 by construction. For those, the ceiling ranks the target only among the nodes of
degree 0, and so removes every training node with an edge of that label. That measures knowing that the answer is an
unseen node, not hubs. 8.11's form did not separate the two.

**The read (zsem2hk2.py, new; no ranking is recomputed):**
- From each R1k npz (sha checked against its record) and the graph, it computes C over:
  - all test edges (gate G-A: this must equal the record's C exactly);
  - the edges whose target is a training node ("seen"), and the rest ("unseen");
  - the seen ones split again by whether the target's degree is above 0.
- It reports each subset's share of the (read group, target) pairs, and does the same per read group.
- A toy check on zsem2hk's own self-test output passed before this was committed: G-A holds, the subsets partition the
  edges, and every unseen target has degree 0.

**What follows.** 8.11 asks for the next hub step on the labels with room. Where that step points depends on where the
room is:
- in seen targets: the popularity of training nodes;
- in unseen targets: the split's unseen nodes, which is an inductive question rather than a hub one.

So the next step is declared after this split is read and before any fit.

**Rows:** sem2hk2-split-<ds> for the seven datasets (e3v), each after its R1k row, with R1k's memory; then
sem2hk2-report. **Outputs:** outputs/sem2/r1k/split/, committed.

### 8.13 Amendment: WebQSP's gate stop, and R1c's remaining rows (24 November 2026, 17:18 IST, 11:48 UTC)

**What happened.** sem2hk-r1k-webqsp-e3v stopped at 8.11's gate G-B:
- Its one-pass raw ranks agreed with zsem2.rank_read's on 99.9609% of the rotation's test edges. The gate asks for
  exact agreement.
- It agreed with R1's saved cosine ranks on 99.7201% of edges. The gate asks for at least 99.9%.
- Before the gate stopped the row, its log printed pooled MRR for the rotation raw and at the ceiling (46.40 and 47.64).
  No MRR_T was computed and no record was written.

**Why.** zsem2hk.rank_ceiling (and zsem2hc.rank_beta) block the nodes by 5e7 // Q; zsem2.rank_read blocks them by
1e8 // Q. The scores are the same products, but different block shapes can round the last bit differently. On WebQSP's
2.6 million nodes, that moves a few scores across the 1e-6 tie band. On the four smaller graphs the ranks were identical.

**The gate stands as declared.** zsem2hkb.py (new) imports zsem2hk and replaces only rank_ceiling, with rank_read's
block size. So the raw ranks are rank_read's own arithmetic in the same process, and G-B can be met exactly.
- Its self-test (zsem2hk's, run through the wrapper) passed.
- A check confirmed that r1k_core calls the replaced function.
- Records written through it carry wrapper_sha256 and block_rule.

**Rows:**
- **R1k:** WebQSP runs again as sem2hk-r1k-webqsp-e3v-b through zsem2hkb.py. HotpotQA's and 2Wiki's rows, not yet
  run, switch to zsem2hkb.py. sem2hk-report and the WebQSP split row point at the new row.
- **R1c (8.10), the six rows not yet run** (WebQSP, HotpotQA and 2Wiki, both arms), are dropped:
  - MetaQA's BELOW already makes 8.10's adjustment NOT_CARRIED, so they cannot change the verdict.
  - zsem2hc.rank_beta has the same block size, so its exact G-B would likely stop them on the large graphs too.
  - sem2hc-report and sem2hc-report-all run over the eight rows read, which list the rest as missing.
