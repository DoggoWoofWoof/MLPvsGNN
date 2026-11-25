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

### 8.14 The query-bank normalization (R1q; 24 November 2026, 17:51 IST, 12:21 UTC), before any R1q number

**What had been read when it was written:**
- R1k (8.11), e3v, C in MRR_T points: MultiHop-RAG +4.93, SQuAD +4.86, MuSiQue +2.16 and HotpotQA +1.13 (ROOM);
  MetaQA +0.90 (no ROOM). WebQSP and 2Wiki had not been read. X9 fails.
- 8.12's post-hoc split on four datasets (it decides nothing). C on unseen targets (test or dev nodes) against seen
  targets (train nodes): MultiHop-RAG +8.52 vs +4.03, SQuAD +11.32 vs +3.21, MuSiQue +5.41 vs +1.33, MetaQA +1.67 vs
  +0.71.
- R1c's report (bd2f51c): NOT_CARRIED.

**Why.** 8.11 asks for the next hub step to be declared before any fit, since some datasets have ROOM. Two things in the
reads point to the same step:
- **The room is two to four times larger on unseen targets.** No training edge touches a test or dev node, so their
  train degree is 0 by construction. A score built on train degree cannot tell a new node from a rare one, and R1c's dev
  choices found no strength that helps.
- **On MetaQA's hub relations, the nodes above the right answer are more popular answers of the same kind** (has_genre:
  +11.7 at the ceiling). A penalty on train degree pushes every genre below the nodes that are no genre at all (R1h:
  -23 on has_genre).

So popularity is measured in the vector space instead: how strongly the operator sends training queries towards a node.
Every node has this number, seen or new. Among nodes of one kind, it is largest for the one most queries land near. With
the full strength this is the inverted softmax (Smith et al., 2017, for orthogonal maps), which query-bank normalization
(Bogolin et al., 2022) uses against hubs in retrieval.

**The method (no fit):**
- R1's saved rotation, as it is.
- **The bank:** for each read group g, up to 4,096 training edges (both ends train nodes), drawn by zsem2.take with
  seed 1 (R1's samples use seed 0). Each bank edge gives the query q_b of its source under its own label's operator.
- **The semantic degree:** A_g(y) = tau log sum_b exp(q_b . x_y / tau), over g's bank queries whose source is not y,
  with tau = 0.05 (R1's). A node that no bank query can count takes the group's smallest value.
- **The score:** q . x_y - alpha_g A_g(y), with alpha_g in {0, 0.25, 0.5, 0.75, 1}, chosen per read group as in 8.10
  (dev MRR_T; a larger alpha must beat every smaller one strictly). alpha = 0 is R1. alpha = 1 is the inverted softmax
  at R1's temperature.
- The test read uses each group's alpha_g.
- **The cosine** gets its own bank and alphas, beside, on MultiHop-RAG, SQuAD, MuSiQue and MetaQA only. On the three
  large graphs its bank pass would double the cost, and it decides nothing.

**The call (per dataset; e3v decides, frozen is read beside on the four small datasets):**
- **H5:** the normalized rotation against R1's rotation raw, on MRR_T, with 8.2's paired bootstrap.
- **Beside** (declared here, so none of these is post hoc):
  - H5 on pooled MRR;
  - H5 on seen targets and on unseen targets, split as 8.12;
  - the normalized rotation against the cosine, and the normalized cosine against the cosine;
  - per group, the semantic degree's mean at seen and at unseen test targets.

**Carry (as 8.9).** The normalization is carried into R2 if, on e3v and MRR_T, H5 is ABOVE on MetaQA and BELOW on none of
the seven datasets.
- If it is carried, R2 uses it, with a bank drawn from R2's own training edges and the alphas chosen again on R2's dev
  sample (same grid, same tie rule).
- If it is not, the one hub step left is the per-query gate that 8.11 named, which needs a fit. It is declared from
  R1q's per-group reads before any fit, or the hub line closes and R2 works on the operators.

**Gates:**
- **G-R** as in 8.10.
- **G-S:** every bank edge has both ends on train nodes.
- **G-B:** at alpha = 0, the one-pass ranks equal zsem2.rank_read's exactly, on dev and on test, for the rotation. They
  use rank_read's block size (8.13). The cosine is gated as in 8.10.
- **G-C:** on the self-test, ranks with a different alpha per query equal a full sort, for the cosine and for a trained
  rotation.
- **G-L:** on the self-test, the blocked semantic degree equals a float64 sum within 1e-4.

**Expectations, written before any number:**
- X10: H5 is ABOVE on SQuAD and on MultiHop-RAG (e3v), where seen targets alone have at least 3 points at the ceiling.
- X11: on SQuAD, H5 is larger on unseen targets than on seen targets.

**Code, rows and outputs:**
- **Code:** outputs/mp_unified/zsem2hq.py (new). It imports zsem2hc (for its dev choice over the same grid), zsem2h and
  zsem2, and edits none of them. Commands: `selftest`, `r1q --ds D --arm A` and `report`. Its self-test passed on the
  laptop before this was committed: G-R, G-S, G-B, G-C and G-L on the toy (largest bank difference 2.4e-7).
- **Rows:**
  - sem2hq-selftest;
  - sem2hq-r1q-multihoprag-e3v first: it is small and checks the real path;
  - then MetaQA, SQuAD and MuSiQue e3v, and the four small frozen rows;
  - then HotpotQA, WebQSP and 2Wiki e3v, each after its R1k row;
  - sem2hq-report after the seven e3v rows, and sem2hq-report-all after all eleven.
- **Outputs:** outputs/sem2/r1q/. The json and md files are committed; the npz files are not.

### 8.15 The hub line closes (24 November 2026, 18:11 IST, 12:41 UTC)

**What had been read when it was written:** R1q (8.14) on MultiHop-RAG, MetaQA, SQuAD and MuSiQue, e3v (fe2f47c).
- Dev chooses alpha = 0 in every read group except MetaQA's R.mention (0.25), and dev MRR_T falls as alpha grows.
- H5 on MRR_T: +0.00 LEVEL on MultiHop-RAG, SQuAD and MuSiQue; MetaQA -0.02 [-0.03, +0.00] LEVEL.
- 8.14's carry rule needs H5 ABOVE on MetaQA. MetaQA is LEVEL, so R1q is not carried, whatever HotpotQA, WebQSP and
  2Wiki read. X10 and X11 fail. zsem2hq's report reads INCOMPLETE until the three large rows run; that label counts
  missing rows only and does not change the call.
- At the true targets the semantic degree is above the median node's in every group, for seen and unseen targets alike.
  Popularity, from training links (R1c) or from the vectors (R1q), goes with being the answer.
- Beside: the rotation beats the cosine on MRR_T on all four (+5.63 MetaQA, +6.29 MuSiQue, +6.69 MultiHop-RAG, +16.04
  SQuAD; all ABOVE). R1's gain holds when each answer counts once, so it is not hub landing.

**The decision (the user, 24 November 2026).** 8.14 left two ways on: declare the fitted per-query gate, or close the
hub line. The user closed it. The per-query gate is not declared, and no hub step will be.

**What this fixes:**
- **No hub step enters R2:** not the hub-balanced fit or the logit adjustment (8.4, 8.3), not R1c's penalty, not R1q's
  normalization, not a per-query gate. R2 uses R1's operator form as R1 trained it (7.9).
- **R1's call is unchanged:** pooled MRR as in 7.7. Its filing reports MRR_T beside it, as 8.10 set.
- **The hub rows still queued only complete records:** R1k on WebQSP and 2Wiki and sem2hk-report; R1q on HotpotQA,
  WebQSP and 2Wiki, the four small frozen R1q rows, sem2hq-report and sem2hq-report-all. None of them can reopen the
  line. R1k is a bound, not a method, and R1q's call is already set by MetaQA. They run on CPU that nothing else is
  using, and R2's rows go ahead of them in the queue.
- **8.12's post-hoc split is not extended:** its WebQSP, HotpotQA and 2Wiki rows are dropped. sem2hk2-report reads the
  four splits that ran (MultiHop-RAG, MetaQA, SQuAD and MuSiQue).

**What comes next:** R2, declared before any R2 number (7.9): the traversal of section 3 with the rotation, on all
seven datasets.

## 9. R2a declared: the traversal in the first stage, parameter-free (24 November 2026, 18:37 IST, 13:07 UTC)

**What had been read when it was written:**
- R1 (section 7): the rotation LEARNS on e3v on all seven datasets, so 7.9's condition for R2 (HOLDS or PARTIAL on
  e3v) is met. The frozen arm LEARNS on the five rows read; its HotpotQA and 2Wiki rows are still training. R1's
  report has not run, and R1 is not filed.
- 8.15: the hub line is closed, so R2 uses R1's rotation as R1 trained it.
- SEM1's first-stage machinery (base, control, recall and bootstrap), which R2 reuses unchanged.
- Code-path smokes only, no read-carve number: the self-test (9.7), and e3v on MultiHop-RAG's dev split (its
  selection carve), with both lambdas fixed at 0.5. 9.9 gives what the smokes changed and what they showed.

**The question.** Does following R1's operators from the first stage's seeds add documents the first stage misses?
And is the gain the operators' and not the cosine neighbourhood's? Nothing is learned in R2a: alpha has no weights,
and lambda is picked from a grid of five on the selection carves.

### 9.1 Vectors, operators, questions and the first stage

- **Arms.** e3v decides, on all seven datasets (the universal rule). The frozen arm is read beside on the four small
  sets (MultiHop-RAG, SQuAD, MuSiQue and MetaQA), as R1q's was; it decides nothing.
- **Operators.** R1's saved rotation for each dataset and arm (outputs/sem2/r1/<ds>/<arm>.npz), loaded as trained:
  no refit, and no weight fit on questions. Gate G-R (9.7) checks that the loaded model is R1's.
  - Labels as R1 planned them: R labels with an operator, followed in either direction; S labels in their stored
    direction, rare S relations through R1's pooled operator. R labels that R1 gave no operator are not followed.
- **Questions and golds:**
  - Selection: s1sel (SEM1's selection carve) on SQuAD, MuSiQue, MetaQA, HotpotQA and 2Wiki; MultiHop-RAG's dev split
    (224 questions).
  - Read: s1eval (metaqa 9,785, squad 11,873, musique 2,417, hotpotqa 7,405, 2wiki 12,576, webqsp 1,503) and
    MultiHop-RAG's test split (1,130).
  - WebQSP has no selection carve. It is read only; its lambdas come from the other six.
- **First stage, as SEM1:** v = zenc's exact dense top 1,000 on the arm's vectors; SPLADE's top 1,000; base =
  RRF(v, SPLADE) with C = 60. MultiHop-RAG has 609 nodes, so its lists end early.

### 9.2 Seeds

The top 5 of v, then the top 5 of SPLADE not already among them: 5 to 10 seeds per question.

### 9.3 The traversal

- x = P v / |P v| for every node (R1's map, then unit length). m_r(s) = T_r x_s at unit length: seed s moved by
  operator r, as R1's query moves a source.
- **Attention, no weights.** ρ = x_q − (x_q · x_s) x_s is the question's residual beyond the seed.
  - α_r(q, s) = softmax over r of (ρ̂ · m_r(s)) / τ, with τ = 0.05, R1's temperature. If |ρ| < 1e-6, α is uniform.
  - The operator chosen is the edge that leads where the question points beyond the seed (9.9 gives why not
    x_q · m_r(s)).
  - mix(s) = Σ_r α_r m_r(s).
- **Hop 1:** score(d) = max over seeds of mix(s) · x_d, plus λ x_q · x_d, over every node except the question's
  seeds.
  - Exact: blocked dense products in the raw space ((mix P) · v_d / |P v_d|). No approximate index.
  - Ties go to the lower node id.
- **The list (1,000 long):** the seeds (hop 0, in 9.2's order), then the hop-1 nodes by score.
- **trav0:** the same traversal with every operator at its start: score(d) = max over seeds of v̂_s · v̂_d + λ q̂ · v̂_d,
  with the same seeds first. It is the cosine neighbourhood of the seeds. T2 credits the operators only beyond it.
- **oracle (a ceiling, not a method: it reads golds):**
  - For a seed with a record edge to one of its question's golds (an R label in either direction, an S label from the
    seed as stored), α is one-hot on the operator, among those edges' labels, whose m_r(s) scores the gold highest.
    Ties go to the lower operator id.
  - Other seeds keep their α. λ is trav's.
  - It bounds what a better choice of operator could give with these operators and seeds.

### 9.4 Selection of λ

- Grid: {0, 0.25, 0.5, 1, 2}, for trav and trav0 separately.
- For each read dataset: the λ with the highest mean selection R@5 of RRF(v, list, SPLADE) over the other selection
  datasets of its arm. Leave-one-dataset-out, each dataset counting once. Ties go to the smaller λ.
- e3v's selection datasets are the six with a selection carve, so WebQSP's λ uses all six. Frozen's are the four
  small sets.

### 9.5 The read

- Per dataset and arm, on the read carve: fused = RRF(v, list, SPLADE), C = 60, top 20, for trav, trav0 and oracle.
- Recall = min(hits, golds) / golds per question, averaged. **R@5 of fused is primary.** R@20 is beside.
- Beside, also:
  - each list alone at R@5 and R@1,000 against v alone;
  - hop 1 alone at R@20;
  - SEM1's control, RRF(v, v, SPLADE);
  - oracle reach: the share of questions with at least one seed that has a record edge to a gold;
  - the mean α per operator, and the mean number of seeds.

### 9.6 The calls

- **T1 = trav − base** and **T2 = trav − trav0**, both on fused R@5.
  - Paired bootstrap over questions: 2,000 resamples, seed 0, 95% interval. ABOVE, LEVEL or BELOW.
- **ADDS** on a dataset = T1 ABOVE and T2 ABOVE.
  - T1 alone cannot credit the operators. Every list starts with the seeds, which repeats v's and SPLADE's tops in
    the fusion; SEM1's control shows the size of that.
  - trav0 has the same seeds first, so T2 isolates the operators.
- **The e3v verdict:**
  - HOLDS if ADDS on all seven;
  - PARTIAL if ADDS on at least one and T1 BELOW on none;
  - FAILS otherwise.
  - INCOMPLETE while a row is missing. That label counts missing rows only.
- **The ceiling:** C = oracle − trav on fused R@5. **ROOM** on a dataset if C ≥ 1 point.
- **Beside, deciding nothing:** trav − control, trav0 − base, T1 and T2 at R@20, and trav alone − v alone at R@1,000.
- **Expectations, written before any number** (a failed one is reported, not explained away):
  - X12: T1 ABOVE on MuSiQue. Its golds are chained paragraphs, and a later hop's gold is one edge from an earlier
    one.
  - X13: T1 not ABOVE on SQuAD. Its answers sit in one paragraph, so there is little to traverse to.
  - X14: ROOM on MetaQA. Its typed relations are many, and a parameter-free α can pick the wrong one.

### 9.7 Gates (a failure stops that row)

- **G-E:** as 7.8.
- **G-R:** R1's test sample, read groups and operators are reproduced. The loaded rotation's test ranks agree with
  R1's saved ranks on at least 99.9% of test edges, with MRR within 0.01.
- **G2 (SEM1's):** base is SEM1's list.
  - The six: hits equal to E0's h_rrf5 on at least 99.5% of questions, and R@5 within 0.0005 of E0's.
  - MultiHop-RAG: the top-5 sets equal MHR1's first-stage lists (outputs/mhr1/first/lists.npz) on at least 99.5%.
- **G-X:** no seed is among its own question's hop-1 nodes.
- **Self-test** (a toy with planted tails: n 2,000, D 32, D' 40, 120 questions, two R labels and two directed S
  labels):
  - G-T: each list equals a float64 full sort on at least 99% of questions.
  - G-X, and G-H0: each list is its seeds, then its hop-1 nodes.
  - G-0: with every operator at its start (D' = D, angles 0, plane weights 1), trav's top 20 equals trav0's on at
    least 99% of questions.
  - Planted: the tail in the top 10 for trav at λ 0 on at least 80% of questions, above trav0, and oracle at or above
    trav.

### 9.8 What follows (fixed now)

- **ROOM on any e3v dataset:** R2b, the learned α of section 3 (the small net on the question against each moved
  seed, the label name's vector and the seed's degree per label).
  - Cross-fitted leave-one-dataset-out, on the training carves' golds, stopped by P5b2's stop.
  - Declared before any fit, and read against R2a's trav. It must close part of C.
- **No ROOM anywhere:** α is not learned, and nothing is fit.
- **HOLDS or PARTIAL:** R3's transfer reads (section 4) are declared on the form that carries (R2a's, or R2b's if it
  is declared and carries).
- **FAILS with no ROOM:** the traversal closes in the first stage, and SEM2 stands on R1 (link recovery).
- No call here touches the journal's layout or pools.

### 9.9 Changes made before this declaration (code-path smokes, not reads)

1. **Attention.** The first version scored x_q · m_r(s). The seed comes from the question's own top list, so x_q is
   close to x_s, and that score favours the operator that moves the seed least. It was replaced by the residual
   (9.3) on reading the code, before any run.
2. **The seeds head the list.** The first version's list held the hop-1 nodes only. In the MultiHop-RAG dev smoke
   (e3v, λ 0.5), the fused R@5 fell from 0.640 (base) to 0.10 (trav0: 0.08).
   - Left out of the list, the seeds lost the third list's vote in the fusion, which every other node could get.
     98.5% of the fused top 5 were hop-1 nodes.
   - The seeds now head every list (9.3), trav0 and oracle included.
   - With that, the same smoke gave: base 0.640, control 0.649, trav 0.650, trav0 0.650.
   - MultiHop-RAG has one operator (R.name_comention), so its α is always 1 and its oracle equals trav.
   - These are selection-carve numbers on one dataset at one λ. The selection rows compute them again under 9.4.
3. **The self-test's toy:** the angles were widened (±U(2, 3)) and the question's weight on the planted residual set
   to 0.3, so that the planted tail is not itself a seed. This changes the test only.

### 9.10 Code, rows and outputs

- **Code:** outputs/mp_unified/zsem2t.py (new). It imports zsem2, zsem2h, zsem1, zap1, zul1, zmhr, zenc, zenc1 and
  zenc3v, and edits none of them. Commands: `selftest`, `sel --ds D --arm A`, `read --ds D --arm A` and `report`.
- **Rows:** host CPU, 8 threads, ahead of the remaining hub rows (8.15):
  - sem2t-selftest;
  - sem2t-sel-<ds>-e3v on the six selection datasets, and sem2t-sel-<ds>-frozen on the four small;
  - sem2t-read-<ds>-<arm>, each after all of its arm's selection rows;
  - sem2t-report.
  - The MultiHop-RAG rows carry MHR1's first-stage lists and SPLADE cache as inputs; neither is on the host.
  - Estimates: minutes per selection row, and 1-3 h for each of the HotpotQA and 2Wiki reads.
- **Outputs:** outputs/sem2/r2/sel/<arm>/<ds>.json, read/<arm>/<ds>.json and .npz, report.json and report.md. The json
  and md files are committed; the npz files are not.

### 9.11 Amendment: memory only (24 November 2026, 23:34 IST, 18:04 UTC), before any WebQSP or HotpotQA read number

**What happened:** the e3v reads on MultiHop-RAG, SQuAD, MuSiQue and MetaQA ran. The HotpotQA and WebQSP reads ran out
of memory and wrote no number.
- **WebQSP:** it has 791 operators. The traversal's batch of 128 questions builds a (128 questions, 10 seeds, 791
  operators, D' 1,408) tensor of 5.7 GB per copy, and the host refused the allocation inside the traversal (9.3) with
  40 GB asked.
- **HotpotQA:** the dense search before the traversal could not get 312 MiB while the host was full (R1's 2Wiki row,
  the WebQSP read and this row together). That is in zenc's search, not in R2.

**What changes (memory only):**
- **WebQSP's read** runs through outputs/mp_unified/zsem2tb.py (new), which sets the traversal's batch to 32 questions
  and calls zsem2t.py's own main. zsem2t.py is not edited, so every record's script_sha256 stays zsem2t.py's.
  - A question's list does not depend on its batch-mates. The MultiHop-RAG dev smoke with batch 32 gave the same
    recall as with 128 on every key.
  - The record carries traversal_batch and wrapper.
  - If another read fails the same way (2Wiki has the next-largest tensors), it takes the same wrapper under a new
    row name, with the batch in its row.
- **WebQSP and HotpotQA ask 48 GB** instead of 40 and 32. The rows run again under new names (sem2t-read-webqsp-e3v-b,
  sem2t-read-hotpotqa-e3v-b). sem2t-report and sem2t-report-all depend on those names.

**What does not change:** the seeds, the operators, the attention, the lists, lambda's selection, the calls, the gates
and the expectations. No R2 number has been read on either dataset.

### 9.12 R2a results (25 November 2026, 08:15 IST, 02:45 UTC): the e3v verdict is PARTIAL, and there is no ROOM anywhere

**What ran:** every read of section 9 on the host, on the CPU: the seven e3v reads and the four frozen reads (MultiHop-RAG, SQuAD, MuSiQue, MetaQA), at the λ that 9.4 chose on the
selection carves (chosen λ for trav and trav0 per dataset in each record). Gates **G-E, G-R, G2 and G-X PASS on all eleven reads**, and every record's script_sha256 is zsem2t.py's
(`7d7cf31987df`). Two reads ran through wrappers, and neither changes a list (a wrapper calls zsem2t.py's own main; the record's script_sha256 stays zsem2t.py's): WebQSP through
zsem2tb.py (9.11: traversal batch 32, record field traversal_batch) on top of zsem2tc.py's checkpointing (4 saves, no resume), and HotpotQA through zsem2tc.py (a running state written every 300 s;
the row sem2t-read-hotpotqa-e3v-b, 48 GB, was replaced by sem2t-read-hotpotqa-e3v-r, 26 GB; the read resumed from question 4,608 of 7,405 and the record carries the checkpoint, 8 saves).
Every read ran with 8 threads. Outputs: outputs/sem2/r2/{sel,read}/..., report.json and report.md (json and md committed, npz not).

**The reads (fused R@5 on the read carve, in %; control = SEM1's RRF(v, v, SPLADE); the calls are the paired intervals of 9.6, in points):**

| arm/dataset | base | control | trav | trav0 | oracle | T1 = trav - base (points, 95% interval) | T2 = trav - trav0 | trav - control | C = oracle - trav | reach | ADDS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e3v/multihoprag | 60.94 | 62.50 | 62.78 | 62.60 | 62.78 | +1.84 [+1.01, +2.65] ABOVE | +0.18 [+0.00, +0.38] LEVEL | +0.28 | +0.00 | 0.997 | no |
| e3v/squad | 88.25 | 87.36 | 88.53 | 88.42 | 88.52 | +0.28 [+0.04, +0.51] ABOVE | +0.11 [+0.03, +0.19] ABOVE | +1.17 | -0.01 | 0.813 | yes |
| e3v/musique | 47.77 | 48.62 | 48.98 | 49.21 | 49.01 | +1.22 [+0.76, +1.67] ABOVE | -0.22 [-0.37, -0.09] BELOW | +0.36 | +0.03 | 0.891 | no |
| e3v/metaqa | 0.45 | 0.59 | 0.56 | 0.50 | 0.58 | +0.11 [+0.04, +0.19] ABOVE | +0.06 [+0.02, +0.11] ABOVE | -0.03 | +0.02 | 0.548 | yes |
| e3v/webqsp | 4.56 | 4.88 | 4.94 | 4.94 | 4.92 | +0.38 [+0.09, +0.72] ABOVE | +0.00 [+0.00, +0.01] LEVEL | +0.06 | -0.02 | 0.925 | no |
| e3v/hotpotqa | 68.17 | 68.05 | 69.14 | 69.06 | 69.12 | +0.97 [+0.71, +1.22] ABOVE | +0.08 [-0.02, +0.18] LEVEL | +1.09 | -0.02 | 0.936 | no |
| e3v/2wiki | 63.97 | 64.19 | 64.62 | 64.59 | 64.65 | +0.65 [+0.51, +0.80] ABOVE | +0.03 [-0.02, +0.08] LEVEL | +0.43 | +0.04 | 0.866 | no |
| frozen/multihoprag | 63.02 | 64.62 | 65.07 | 64.85 | 65.07 | +2.04 [+1.21, +2.87] ABOVE | +0.21 [-0.05, +0.50] LEVEL | +0.44 | +0.00 | 0.998 | no |
| frozen/squad | 90.53 | 90.30 | 90.98 | 90.90 | 90.99 | +0.45 [+0.21, +0.67] ABOVE | +0.08 [-0.02, +0.18] LEVEL | +0.68 | +0.01 | 0.824 | no |
| frozen/musique | 47.34 | 47.64 | 48.56 | 48.43 | 48.61 | +1.22 [+0.77, +1.67] ABOVE | +0.13 [-0.08, +0.34] LEVEL | +0.92 | +0.05 | 0.894 | no |
| frozen/metaqa | 0.47 | 0.76 | 0.70 | 0.68 | 0.70 | +0.22 [+0.15, +0.30] ABOVE | +0.01 [-0.04, +0.07] LEVEL | -0.06 | +0.00 | 0.545 | no |

**The calls (9.6):**
- **The e3v verdict is PARTIAL** (ADDS on at least one dataset, T1 BELOW on none): **ADDS on SQuAD and MetaQA only** (T1 and T2 both ABOVE). T1 is ABOVE on all 11 of 11 reads, but T1 alone cannot credit the operators:
  T1 is the traversal against the base list, and trav0 (the cosine neighbourhood of the same seeds, no operator, 9.3) is already most of it. **trav0 - base** over the seven e3v datasets (MultiHop-RAG, SQuAD, MuSiQue, MetaQA,
  WebQSP, HotpotQA, 2Wiki) is +1.66, +0.17, +1.44, +0.05, +0.38, +0.89, +0.61 points, against T1's +1.84, +0.28, +1.22, +0.11, +0.38, +0.97, +0.65. T2 = trav - trav0 is what the operators add: **ABOVE on SQuAD and MetaQA, LEVEL on MultiHop-RAG, WebQSP, HotpotQA and 2Wiki, and BELOW on MuSiQue**
  (-0.22 [-0.37, -0.09] BELOW: on that dataset the operators move the seeds' neighbourhood the wrong way, a little).
  Beside the SEM1 control, **trav - control** is ABOVE on SQuAD, HotpotQA and 2Wiki and LEVEL on the other four e3v datasets (-0.06 to +1.17 points over all eleven reads).
- **The effect sizes are small.** The two ABOVE T2s are +0.11 [+0.03, +0.19] ABOVE on SQuAD and +0.06 [+0.02, +0.11] ABOVE on MetaQA; the other five e3v T2s lie between -0.22 and +0.18.
  T1, the first-stage R@5 gain of the whole traversal over the base, is +0.11 (MetaQA) to +1.84 (MultiHop-RAG). The scales differ: MetaQA's R@5 is 0.45 in the base and
  0.56 after the traversal (WebQSP 4.56 and 4.94), against SQuAD's 88.25 and 88.53.
- **ROOM: none.** C = oracle - trav is -0.02 to +0.04 points on the seven e3v datasets, LEVEL on every one (every interval includes 0 or ends at it), and at most 0.05 on any of the eleven reads
  (frozen MuSiQue, +0.05 [+0.01, +0.11] ABOVE, still twenty times below ROOM's 1-point line). Choosing the best operator per seed, which is what a learned α would do, buys at most hundredths of a point beyond the parameter-free α.
  The oracle keeps the traversal's seeds and operators, so it is bounded by their **reach**, the share of questions with a seed that has a record edge to a gold: 0.548 (MetaQA) to 0.997 (MultiHop-RAG);
  C is also about 0 where reach is high (MultiHop-RAG, WebQSP, HotpotQA, MuSiQue), so the choice of operator is not what limits these lists.
- **Beside, deciding nothing (9.6): the operators show deeper in the list.** T2 at R@20 (what the operators add at 20 candidates instead of 5) is ABOVE on MuSiQue (+0.74 [+0.36, +1.13] ABOVE), HotpotQA (+0.47 [+0.30, +0.63] ABOVE) and 2Wiki (+0.31 [+0.23, +0.40] ABOVE), LEVEL on the other four e3v datasets, BELOW on none.
  The traversal's list alone against v's alone at R@1,000 is ABOVE on 2Wiki only (+9.41 [+9.04, +9.76] ABOVE), LEVEL on MultiHop-RAG and HotpotQA, and BELOW on SQuAD (-0.93), MuSiQue (-1.06), MetaQA (-9.53) and WebQSP (-3.61).
  These reads call nothing and no rule depends on them; they say where an operator effect, if there is one, shows: below the top 5, on MuSiQue, HotpotQA and 2Wiki.
- **Expectations:** X12 (T1 ABOVE on MuSiQue) **held** (+1.22 [+0.76, +1.67] ABOVE). **X13 (T1 not ABOVE on SQuAD) failed**: T1 is +0.28 [+0.04, +0.51] ABOVE, and SQuAD is one of the two ADDS datasets, by +0.11 points of T2.
  **X14 (ROOM on MetaQA) failed**: C is +0.02.
- **The frozen arm** (the four small datasets) agrees in sign: T1 ABOVE on all four, T2 LEVEL on all four, ROOM none (C is within 0.05 points on all four); no verdict is drawn on it (9.6 draws it on e3v).

**What follows (9.8, fixed before the numbers):**
- **No ROOM anywhere: α is not learned and nothing is fit.** R2b is not declared.
- **PARTIAL: R3's transfer reads (section 4) are declared on R2a's form,** which is the form that carries. They are not declared here, and no R3 number exists.
- The reading of this section, said once: the parameter-free traversal's first-stage gain over the base is mostly the seeds' own cosine neighbourhood (trav0), which needs no operator; the operators add +0.11 and +0.06
  points on two datasets at R@5, nothing measurable on four, and lose 0.22 on one; the beside reads at R@20 show an operator effect on three datasets (MuSiQue, HotpotQA, 2Wiki), which no call rests on;
  and a better choice of operator has no room (C). R2a gives no reason to learn α. No call here touches the journal's layout or pools.

### 8.16 Records completed after the closure (24 November 2026, 23:47 IST, 18:17 UTC): decides nothing

8.15 closed the hub line with five rows still running. They have finished: R1k on WebQSP and 2Wiki, and R1q on
HotpotQA, 2Wiki and WebQSP, with the reports over all seven datasets. Nothing here reopens the line, and no hub step
follows.

**R1k, the hub ceiling (8.11; e3v, MRR_T in points, ceiling minus rotation):**

| dataset | C | verdict |
| --- | --- | --- |
| MultiHop-RAG | +4.93 [+4.06, +5.81] | ROOM |
| SQuAD | +4.86 [+4.33, +5.43] | ROOM |
| MuSiQue | +2.16 [+1.85, +2.50] | ROOM |
| MetaQA | +0.90 [+0.78, +1.02] | NO_ROOM |
| WebQSP | +0.68 [+0.57, +0.78] | NO_ROOM |
| HotpotQA | +1.13 [+0.94, +1.32] | ROOM |
| 2Wiki | +1.30 [+1.09, +1.50] | ROOM |

The two new rows follow the pattern of the others: room exists on the text-linked graphs, and the ceiling is small on the
two knowledge-base graphs. 8.12's seen/unseen split stays on the four datasets it ran on.

**R1q, the query-bank normalization (8.14; e3v, H5 on MRR_T):**
- H5 is LEVEL on HotpotQA (-0.05 [-0.16, +0.05]), and exactly 0.00 on 2Wiki and WebQSP, where the dev sample picks alpha
  0 in every group.
- MetaQA: -0.02 [-0.03, +0.00] LEVEL on the test groups and -0.04 [-0.07, -0.01] BELOW pooled; the frozen arm is BELOW
  (-0.05 [-0.10, -0.02]).
- The carry rule (8.9 form: ABOVE on MetaQA and BELOW on none) reads NOT_CARRIED. X10 and X11 are both False, as 8.15
  said on the four small datasets.

**Files** (json and md only; the npz stay on the host): outputs/sem2/r1k/2wiki/e3v.json, outputs/sem2/r1k/webqsp/e3v.json,
outputs/sem2/r1q/hotpotqa/e3v.json, outputs/sem2/r1q/2wiki/e3v.json, outputs/sem2/r1q/webqsp/e3v.json, and the reports
outputs/sem2/r1k/report.*, outputs/sem2/r1k/split/report.* and outputs/sem2/r1q/report.*. The laptop's rerun of each
report matches the host's except the time stamp.

### 7.12 R1 filed (24 November 2026, 23:57 IST, 18:27 UTC)

All fourteen rows ran (seven datasets, two vector sets) and the report has no missing row. The laptop's rerun of the
report matches the host's except the time stamp. Declared in section 7; the calls are 7.7's.

**R1: link recovery on held-out nodes (MRR in points, cosine to rotation; delta = rotation - cosine, paired 95%
bootstrap over test edges; MRR_T beside, never called on, per 8.10):**

| dataset | e3v MRR | e3v delta, call | e3v MRR_T delta | frozen MRR | frozen delta, call | frozen MRR_T delta |
| --- | --- | --- | --- | --- | --- | --- |
| MultiHop-RAG | 17.36 to 24.12 | +6.76 [+5.16, +8.28] LEARNS | +6.69 [+4.81, +8.56] | 18.89 to 24.09 | +5.20 [+3.59, +6.81] LEARNS | +5.17 [+3.20, +7.05] |
| SQuAD | 12.04 to 28.28 | +16.23 [+14.95, +17.54] LEARNS | +16.04 [+14.64, +17.41] | 10.19 to 28.49 | +18.30 [+16.97, +19.68] LEARNS | +18.61 [+17.22, +20.00] |
| MuSiQue | 5.7 to 12.47 | +6.77 [+6.00, +7.60] LEARNS | +6.29 [+5.50, +7.14] | 6 to 13.87 | +7.87 [+7.04, +8.74] LEARNS | +7.37 [+6.52, +8.23] |
| MetaQA | 3.37 to 15.68 | +12.31 [+11.74, +12.92] LEARNS | +5.63 [+5.11, +6.14] | 2.79 to 16.43 | +13.65 [+13.02, +14.28] LEARNS | +7.01 [+6.48, +7.56] |
| WebQSP | 7.32 to 46.4 | +39.07 [+38.36, +39.75] LEARNS | +35.43 [+34.42, +36.38] | 5.19 to 46.24 | +41.06 [+40.31, +41.75] LEARNS | +40.02 [+38.98, +41.05] |
| HotpotQA | 7.31 to 10.95 | +3.64 [+3.13, +4.17] LEARNS | +3.61 [+3.06, +4.14] | 7.46 to 13.08 | +5.61 [+5.01, +6.25] LEARNS | +5.55 [+4.91, +6.21] |
| 2Wiki | 6.74 to 9.36 | +2.62 [+2.11, +3.14] LEARNS | +2.58 [+2.07, +3.09] | 6.64 to 10.96 | +4.32 [+3.72, +4.87] LEARNS | +4.25 [+3.66, +4.83] |

- **R1 per arm: e3v HOLDS (7 of 7 LEARN); frozen HOLDS (7 of 7 LEARN).** Every interval sits above zero. The MRR_T
  reading agrees on every row.
- **Expectations:** X1 held (e3v LEARNS on MetaQA and WebQSP). X2 held (MuSiQue). X3 held (the report's call:
  rotation with D' = D + 256 is not below rotation with D' = D on the small sets). **X4 failed:** frozen's delta is below e3v's on only 1 of 7
  datasets (MultiHop-RAG), not on most. The weaker vector set leaves more to learn, not less.
- **What the operators are doing (beside, not called):**
  - On MultiHop-RAG, whose links carry one label, a shared map without rotations (map_only) recovers the whole gain
    (+6.93 against rotation's +6.76). There is no relation to tell apart.
  - On MetaQA (10 labels) map_only reaches +2.98 and rotation +12.31. On MuSiQue
    (3 labels) it is +3.34 and +6.77. The typed operators carry the gain where the graph is typed.
  - Translation is close to rotation: ahead of it on MetaQA and WebQSP (and by 0.05 on MultiHop-RAG), behind it on
    SQuAD, MuSiQue, HotpotQA and 2Wiki.
- **Sizes:** the gain is large on the knowledge-base graphs (WebQSP and MetaQA, with hits@100 from 21 to 73 percent
  and from 12 to 44 percent on e3v) and small on HotpotQA and 2Wiki.
- **What R1 does not say:** it recovers links among documents. It does not say a question's first stage improves. That
  is R2a (section 9).

**Files:** outputs/sem2/r1/*/e3v.json and frozen.json (fourteen), outputs/sem2/r1/report.json and report.md,
outputs/sem2/r1h/hubread.json and hubread.md (all fourteen rows; it replaces the eleven-row version). The npz stay on
the host.
