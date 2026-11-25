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

## 10. R2c declared: the traversal at the depth a pool takes its seeds from (25 November 2026, 15:11 IST, 09:41 UTC), after R2a's R@20 means were seen

### 10.1 Why, and what was seen before this page

- **Why.** The user, 25 November: first the deeper read of the traversal, then R3 on whichever form survives. The pools of the PL rounds take their seeds from the top of the fused first-stage list
  (PL3's gain on 2Wiki came from widening the seeds to 20). If the traversal changes which golds sit in the top 20, a pool could start from them. This page asks that question of R2a's own records.
- **Seen before this page, so this is declared after them:**
  - 9.12's beside reads at R@20: T2 at R@20 ABOVE on MuSiQue (+0.74), HotpotQA (+0.47) and 2Wiki (+0.31), LEVEL on the other four, and the R@20 means (base, trav0, trav) of all seven e3v records.
    outputs/sem2/r2/report.json also holds T1 at R@20 and T2 at R@20 on every row, as beside entries.
  - **MuSiQue's ALL at 20** (the share of questions with every gold in the top 20): while checking that the stored arrays hold what this rule needs I read its four means (base 24.9, trav0 25.4,
    trav 26.8, oracle 27.2 percent). No interval was computed, and **the other six datasets' ALL at 20 were not read**.
  - λ was chosen on R@5 (9.4). **It is not re-chosen on R@20 here**; a different λ would need new selection reads and new reads on the host.
  - So the call at 20 moves from "beside" to "called" after the R@20 means were seen. ALL at 20 is new (MuSiQue's four means glimpsed).
- **A limit, said first.** A first-stage ALL at 20 is not a pool's coverage. PL3's diagnostic D1 (post-hoc, 300 dev questions) found 656 of the 658 gold nodes missing from the MuSiQue, MetaQA and WebQSP pools inside
  the uncapped three-hop reach of their seeds: those pools lose golds to the caps and the node bound, not to the first stage. Part A cannot say whether a pool improves; it says whether the seeds do.

### 10.2 Part A: the calls (a laptop read of stored records; no host job, no traversal, no fit)

- **Rows:** the seven e3v reads of R2a decide (outputs/sem2/r2/read/e3v/*.npz); the four frozen reads are read beside (no verdict).
- **Two measures at 20, per question, from the stored arrays:**
  - **recall at 20:** min(hits, golds) / golds of the fused list RRF(v, list, SPLADE) at 20 (R2a's r_*_fused20);
  - **ALL at 20:** 1 where that recall is 1 (every gold is in the top 20), 0 where not; a question with no gold is left out.
- **Calls, on each measure:** T1 = trav - base and T2 = trav - trav0, each by the paired bootstrap of 9.6 (2,000 resamples, seed 0, 95% interval; ABOVE, LEVEL or BELOW), through the same function as R2a's report (zsem1.boot).
  **ADDS at 20** on a dataset = T1 and T2 ABOVE on that measure.
- **The e3v verdict at 20, per measure:** HOLDS if ADDS on all seven; PARTIAL if ADDS on at least one and T1 BELOW on none; FAILS otherwise; INCOMPLETE while a row is missing (9.6's rule).
- **Beside, deciding nothing:** trav - SEM1 control, trav0 - base, and C = oracle - trav at 20 (ROOM at 20 if C is 1 point or more). A ROOM at 20 would not reopen R2b: R2b stays undeclared until a page declares it.
- **Rows that cannot move:** on MetaQA and WebQSP the golds are knowledge-base nodes and R@20 is 1.4 and 9.7 percent, so ALL at 20 is near zero there; they are reported and may be LEVEL.

### 10.3 What follows (fixed now, before any number)

- **ADDS on ALL at 20 on at least one e3v dataset: Part B is NAMED.** Part B is a read at the pool level: the traversal's fused list as the seeds of a PL walk. It needs the lists, and R2a's npz keep only the
  top 5; so it is a new host read of the seven datasets through a new wrapper that saves the fused top 50, sized from R2a's records at about 18,000 s of 8-thread host time (2Wiki 11,914, HotpotQA 3,746, WebQSP 1,369,
  SQuAD 437, MetaQA 358, MuSiQue 139, MultiHop-RAG 12), and the walk is a PL round, which the PL engines (pinned) do not take as an option today. **It is not declared here, and no host row is queued for it before the user is told that price.**
- **No ADDS on ALL at 20 anywhere: Part B is CLOSED.** The traversal does not move the seeds' coverage at 20, no host re-read is made, and nothing else changes.
- **Either way R3's transfer reads (section 4) are declared next, on R2a's form.** R2c changes the depth of a read, not the form.
- No call here touches the journal's layout or pools.

### 10.4 Expectations, written before any call (a failed one is reported, not explained away)

- **X15:** ADDS on recall at 20 on MuSiQue, HotpotQA and 2Wiki (partly seen: T2 at R@20 ABOVE there in 9.12; T1 at R@20 is in report.json), not on SQuAD (trav0 and the base are already at 95).
- **X16:** ADDS on ALL at 20 on MuSiQue (its four means were glimpsed: trav 26.8 against trav0 25.4), and not on MetaQA, WebQSP or SQuAD.
- **X17:** Part B is NAMED, through MuSiQue alone or with HotpotQA and 2Wiki.

### 10.5 Gates (a failure stops the report)

- **G-R2:** each record's script_sha256 is zsem2t.py's (`7d7cf31987df`) and each npz hashes to the record's npz_sha256.
- **G-A:** the stored arrays r_base20, r_trav_fused20 and r_trav0_fused20 average to the record's recall values (base20, trav_fused20, trav0_fused20) to 1e-6.
- **Self-test (planted toy):** ALL at 20 from recall 1 only (NaN stays NaN); a planted gain on 15 percent of 400 questions is ABOVE with ADDS, a flat arm is LEVEL without it; the verdicts HOLDS, PARTIAL, FAILS and INCOMPLETE come out as 9.6 says.

### 10.6 Code and outputs

- **outputs/mp_unified/zsem2tr.py** (committed with this page; imports zsem2t.py and zsem1.py, edits neither): `selftest`, `report`. Outputs: outputs/sem2/r2/r2c/report.json and report.md (committed; npz are R2a's).
- The numbers are filed in 10.7 after the commit of this page.

### 10.7 R2c Part A results (25 November 2026, 15:13 IST, 09:43 UTC): PARTIAL on both measures, ADDS on MuSiQue, HotpotQA and 2Wiki, Part B NAMED

**What ran:** `python outputs/mp_unified/zsem2tr.py report` on the laptop (a few seconds; the code is the commit of section 10, c6c71c2f; no traversal, no host job). Gates **G-R2 and G-A PASS on all eleven rows**
(every record's script_sha256 is zsem2t.py's, every npz hashes to its record, the stored r_base20, r_trav_fused20 and r_trav0_fused20 average to the records' values). Outputs: outputs/sem2/r2/r2c/report.json and report.md.

**The calls (e3v decides; points; paired bootstrap of 9.6; the frozen four are in report.md, beside):**

Recall at 20 (min(hits, golds) / golds of the fused list at 20, in percent):

| e3v dataset | base | trav0 | trav | oracle | T1 = trav - base | T2 = trav - trav0 | C = oracle - trav | ADDS (recall at 20) |
|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | 91.54 | 93.48 | 93.73 | 93.73 | +2.19 [+1.53, +2.92] ABOVE | +0.25 [-0.16, +0.67] LEVEL | +0.00 [+0.00, +0.00] LEVEL | no |
| SQuAD | 95.28 | 95.33 | 95.19 | 95.23 | -0.08 [-0.32, +0.16] LEVEL | -0.13 [-0.30, +0.03] LEVEL | +0.04 [-0.02, +0.10] LEVEL | no |
| MuSiQue | 60.42 | 60.70 | 61.44 | 61.88 | +1.02 [+0.45, +1.57] ABOVE | +0.74 [+0.36, +1.13] ABOVE | +0.43 [+0.17, +0.73] ABOVE | yes |
| MetaQA | 1.25 | 1.45 | 1.38 | 2.07 | +0.13 [+0.01, +0.25] ABOVE | -0.07 [-0.19, +0.05] LEVEL | +0.69 [+0.54, +0.84] ABOVE | no |
| WebQSP | 8.34 | 9.42 | 9.68 | 9.60 | +1.33 [+0.70, +2.00] ABOVE | +0.26 [-0.16, +0.71] LEVEL | -0.08 [-0.40, +0.21] LEVEL | no |
| HotpotQA | 77.59 | 77.73 | 78.20 | 78.18 | +0.61 [+0.34, +0.90] ABOVE | +0.47 [+0.30, +0.63] ABOVE | -0.01 [-0.14, +0.11] LEVEL | yes |
| 2Wiki | 69.98 | 70.52 | 70.83 | 71.10 | +0.85 [+0.69, +1.00] ABOVE | +0.31 [+0.23, +0.40] ABOVE | +0.27 [+0.19, +0.34] ABOVE | yes |

ALL at 20 (the share of questions with every gold in the fused top 20, in percent):

| e3v dataset | base | trav0 | trav | oracle | T1 = trav - base | T2 = trav - trav0 | C = oracle - trav | ADDS (ALL at 20) |
|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | 78.05 | 83.89 | 84.69 | 84.69 | +6.64 [+4.87, +8.50] ABOVE | +0.80 [-0.35, +1.95] LEVEL | +0.00 [+0.00, +0.00] LEVEL | no |
| SQuAD | 95.28 | 95.33 | 95.19 | 95.23 | -0.08 [-0.32, +0.16] LEVEL | -0.13 [-0.30, +0.03] LEVEL | +0.04 [-0.02, +0.10] LEVEL | no |
| MuSiQue | 24.91 | 25.45 | 26.81 | 27.18 | +1.90 [+0.99, +2.81] ABOVE | +1.37 [+0.70, +2.03] ABOVE | +0.37 [-0.08, +0.87] LEVEL | yes |
| MetaQA | 0.91 | 1.00 | 0.98 | 1.46 | +0.07 [-0.04, +0.18] LEVEL | -0.02 [-0.13, +0.09] LEVEL | +0.48 [+0.34, +0.61] ABOVE | no |
| WebQSP | 5.92 | 7.12 | 7.05 | 6.99 | +1.13 [+0.53, +1.80] ABOVE | -0.07 [-0.47, +0.33] LEVEL | -0.07 [-0.33, +0.20] LEVEL | no |
| HotpotQA | 57.61 | 57.95 | 58.74 | 58.69 | +1.13 [+0.62, +1.66] ABOVE | +0.80 [+0.49, +1.09] ABOVE | -0.05 [-0.30, +0.19] LEVEL | yes |
| 2Wiki | 38.92 | 39.84 | 40.32 | 40.79 | +1.40 [+1.11, +1.69] ABOVE | +0.49 [+0.32, +0.64] ABOVE | +0.47 [+0.31, +0.62] ABOVE | yes |

- **The e3v verdict at 20 is PARTIAL on recall and PARTIAL on ALL.** ADDS (T1 and T2 ABOVE) on **MuSiQue, HotpotQA and 2Wiki, on both measures**; not on MultiHop-RAG, SQuAD, MetaQA or WebQSP. **T1 is BELOW on none** of the seven, on either measure.
- **The effect sizes are small.** T2 on ALL at 20 (what the operators add at 20 candidates) is +1.37 points on MuSiQue (base 24.91), +0.80 on HotpotQA (base 57.61) and +0.49 on 2Wiki (base 38.92); T1 on ALL is +1.90, +1.13 and +1.40.
  The seeds' own cosine neighbourhood (trav0 - base, difference of means on ALL at 20) is +0.54, +0.34 and +0.91 on those three, and +5.84 on MultiHop-RAG, where T2 is LEVEL (+0.80 [-0.35, +1.95]): there the whole gain is the neighbourhood and no operator is needed.
- **ROOM at 20: none.** C = oracle - trav is below 1 point on all seven e3v datasets on both measures (the largest are MetaQA, +0.69 on recall and +0.48 on ALL), so a better choice of operator still has no room at 20.
- **Expectations: X15 held** (ADDS on recall at 20 on MuSiQue, HotpotQA and 2Wiki, not on SQuAD). **X16 held** (ADDS on ALL at 20 on MuSiQue, and not on MetaQA, WebQSP or SQuAD). **X17 held** (Part B is NAMED, through MuSiQue, HotpotQA and 2Wiki).
- **The frozen four, beside, no verdict:** ADDS on ALL at 20 on MuSiQue (T2 +1.78) and MetaQA (T2 +0.42); none on MultiHop-RAG or SQuAD.
- **Said once, as in 9.12:** λ was chosen on R@5, not on R@20; the R@20 means and three T2 intervals were seen before section 10 (10.1); and ALL at 20 is a first-stage measure, not a pool's coverage.
  On 2Wiki and HotpotQA, whose pools already reach the target (PL3: 2Wiki 0.961 at 917 nodes), a better seed list could make a pool cheaper, not more complete; on MuSiQue the pool is cap-bound (D1), not seed-bound.

**What follows (10.3, fixed before the numbers): Part B is NAMED, and it is not declared or queued.** It is a new host read of the seven datasets that saves the fused top 50 (about 18,000 s of 8-thread host time, sized from R2a's records)
and then a PL round that takes the traversal's list as its seeds (the PL engines are pinned). The gains above are 0.5 to 1.9 points of ALL at 20 on three datasets; the user is told the price and the default is to leave Part B queued last.
R3's transfer reads (section 4) are declared next, on R2a's form. No call here touches the journal's layout or pools.

## 11. R3(c) declared: an operator for a relation never seen, from its name alone (25 November 2026, 15:21 IST, 09:51 UTC), before any R3 number

### 11.1 Why this is the R3 that is left, and what was seen

- **The user, 25 November:** "yeah sure then we can put it to a generalization test" (after being told that R3 as section 4 wrote it has one part left).
- **Section 4's R3, read against what is on file:**
  - **(a)** the left-out dataset "with alpha fit without it": **moot.** R2b was not declared (9.12: no ROOM), so alpha has no weights; the only fitted things are R1's operators, per corpus, and lambda.
  - **(b)** another encoder's vectors with the operators refit from the links and alpha unchanged: **largely read already.** It is R2a's frozen arm (R1: 7 of 7 LEARN on frozen; R2a: T1 ABOVE and T2 LEVEL on all four small datasets at R@5). No new read.
  - **(c)** unseen relations through the name net: **not on file.** Section 3 sketches it ("for a label never seen in training, the angles come from the label name's frozen vector through a small shared net") and no code or page defines it. This page does.
- **Seen before this page:** R1's records (7.12: per-label deltas on the read labels of MetaQA and WebQSP, per-label MRRs of the free-angle operators) and R2a's. The hold-out below is drawn by a hash of the relation's name, blind to all of them.
- **Not seen:** any score of any operator produced from a name. No name has been encoded for this purpose. No R3 number exists.

### 11.2 The question

R1 fits one free set of angles per relation from that relation's own edges. A KB of a new domain brings relations nobody fitted. **Can a small shared net give such a relation an operator from its name alone, and does that operator rank the relation's true tails above plain cosine, and above what a relation-blind operator does?**

### 11.3 Rows

- **MetaQA and WebQSP, on e3v.** They are the two datasets whose S relations are many and named (MetaQA 7 with an operator of their own; WebQSP 791 operators, nearly all S). 2Wiki and HotpotQA carry one S relation and MuSiQue and SQuAD none (7.2), so there is nothing to hold out there.
- **frozen is not read** (R2a's rule: e3v decides). **No other dataset or arm is read.**
- **Relations that can be held out:** the S relations with an operator of their own in R1 (at least 500 training edges, 7.2). S.rare's relations and the R labels are never held out: they are always seen.

### 11.4 The name net

- **Name string of a relation:** the label's name without its family prefix, lower case, "_" and "." read as spaces: S.directed_by is "directed by"; WebQSP's "S.award honor award winner" is "award honor award winner"; the R labels are "mention", "name comention" and "title comention".
- **Name vector:** the string encoded by the e3v encoder as each part encodes a document (the same function that made the stored node vectors: zenc's load_model and encode_texts, each part at its own token budget, fused and scaled as e3v), then divided by its norm. **Gate G-N:** the same function reproduces 64 stored node vectors from their texts, cosine at least 0.999 each (a CPU float32 encode is allowed if it passes; the stored vectors are float16 GPU).
- **The net g:** name vector (D = 1,152) -> 256 hidden units (tanh) -> D'/2 = 704 angles. **The last layer starts at zero**, so at the start every relation's angles are 0 and every score is the cosine (gate G-0, as 7.8). Angle theta_r = g(name_r); the operator is R1's rotation, T_r in D' = D + 256 with block-diagonal 2 x 2 planes. **Directed (S) relations:** score = (T_r x_s) . x_d; **undirected (R):** the symmetric part (7.4).
- **The shared map P** is as R1's (D' = D + 256, start [I; 0] plus N(0, 1e-4^2) extra rows). **No relation has free angles in this fit: seen and unseen relations alike get theirs from g(name).** Parameters: P, and g's two layers (about 0.48 million weights).

### 11.5 The fit (links only; no question, no gold)

- **Relations in three groups, by u = the first 8 bytes of sha256("sem2r3|<dataset>|<label name>") over 2^64:**
  - **WebQSP:** the S relations eligible in 11.3 with **u < 0.20 are test (held out), 0.20 <= u < 0.30 are dev relations**, the others train. The R labels and S.rare are train. One fold.
  - **MetaQA:** its 7 eligible S relations, sorted by u, take fold = rank mod 3 (3 folds of 3, 2 and 2 relations). In each fold the fold's relations are test; of the others, **the one with the smallest u is the dev relation**; the rest, with R.mention and R.name_comention, train. Every eligible relation is held out exactly once.
- **Training edges:** R1's (7.3): the node-inductive split and its train edges, **of train relations only.** Loss, negatives, batch, optimiser, learning rate (1e-3), temperature (0.05) and seed (0) are R1's (7.5); an epoch is up to 200,000 train edges with an equal share per train relation (a relation with fewer gives all it has).
- **Stop:** after each epoch, the dev MRR of the **dev relations'** dev edges (7.3's dev split, up to 500 edges per dev relation, under g(name)). Two epochs without a gain, or 10 at most; the best epoch is kept. **No test relation's edge enters the fit or the stop.** (R1 stopped on seen relations' dev edges; here the stop is on relations the fit never saw, as P5b2's did on new golds.)
- **Gate G-S:** every positive of every batch has a train relation (checked on the sampled indices), and the dev and test relations' names are not among the train relations' name strings.

### 11.6 The read

- **Test edges:** for each test relation, up to **200** test edges (7.3: source a test node), numpy default_rng(0) without replacement. Rank of the true tail among all n nodes, filtered as 7.6 (the source and its other true r-tails in any split removed; ties count half). MRR, hits@10, hits@100.
- **Scorers on the same edges:**
  - **cosine** (v_s . v_d): the baseline;
  - **name net** g(name_r): what is called;
  - **mean name:** the same net on the mean of the train relations' name vectors, the same operator for every test relation (a relation-blind operator the net itself produced);
  - **shuffled name** (beside): each test relation takes another test relation's name (a derangement, seed 0);
  - **free angles** (beside, a ceiling): R1's e3v fit, which saw these relations' edges, on the same edges, if the stored model loads and reproduces its record's rank on the sample (WebQSP's e3v model is not on the laptop; R2a loaded it on the host).

### 11.7 The calls (per dataset)

- **d_r = MRR(name net) - MRR(cosine)** on test relation r's edges; **e_r = MRR(name net) - MRR(mean name).** Each test relation counts once.
- **Delta = the mean of d_r over the dataset's test relations; Gain from the name = the mean of e_r.** Interval for each: **a bootstrap over relations** (2,000 resamples of the test relations, seed 0, 95%), not over edges: a relation is the unit that must generalize. ABOVE, LEVEL or BELOW as 9.6.
- **TRANSFERS** if Delta is ABOVE and the Gain from the name is ABOVE. **GENERIC** if Delta is ABOVE and the gain from the name is not (the shared map and a relation-blind operator carry it; the name adds nothing). **LOSES** if Delta is BELOW. **NO** otherwise.
- **R3(c) verdict:** HOLDS if TRANSFERS on both datasets; PARTIAL if TRANSFERS on one; FAILS otherwise. INCOMPLETE while a row is missing.
- **Beside, deciding nothing:** each relation's d_r with its own interval over edges; how many test relations have d_r > 0; hits@10 and hits@100; shuffled-name against cosine and against the name net; and **the share of the free-angle gain the name recovers** = (MRR(name net) - MRR(cosine)) / (MRR(free) - MRR(cosine)), pooled.
- **MetaQA has 7 test relations in all, so its interval is wide by construction; a LEVEL there is a statement about power, and is read so.**

### 11.8 Expectations, written before any number (a failed one is reported, not explained away)

- **X18:** WebQSP Delta ABOVE. Its relations come in families whose names share words (award honor award winner, award honor award nominee ...), and a shared net can place a new member near its siblings.
- **X19:** WebQSP Gain from the name ABOVE (the name carries more than a relation-blind operator).
- **X20:** MetaQA Delta not ABOVE: three to five train relations and two R labels are too few to learn a map from names to angles.
- **X21:** shuffled name below the name net on WebQSP (the right name matters).

### 11.9 Gates (a failure stops the row)

- **G-N, G-0, G-S** as above. **G-0:** with g's last layer at zero, every test relation's score equals the cosine on 99% of the sampled edges' ranks (7.8's check).
- **G-E:** each family's pair count equals the record's (7.8), as R1.
- **Self-test (toy with planted names):** a graph of 600 nodes and 40 relations in 8 families whose angles are a function of the family's name vector; the fit on 70% of the relations beats cosine on the held-out ones, the mean-name operator does not, the shuffled name is below the right name, and G-0 and G-S hold.

### 11.10 What follows (fixed now)

- **TRANSFERS on at least one dataset:** an operator for a new relation needs no fit, and the claim of SEM2's generalization is **read** (not carried to a layout). A cross-dataset reading (a net fit on WebQSP's relations, applied to MetaQA's) is named, not declared.
- **GENERIC or NO or LOSES on both:** the operators are fit per corpus from its own links, as R1 says, and the name net adds nothing; SEM2's transfer claim stands on R1 (link recovery per corpus) and R2a's frozen arm.
- No call here touches the journal's layout or pools. Nothing is fit on a question.

### 11.11 Code, rows and outputs

- **outputs/mp_unified/zsem2r3.py** (new; imports zsem2.py and zsem2t.py and edits neither): `names` (the encode), `fit`, `read`, `report`, `selftest`. It is committed after this page and before any row runs; a code commit changes no rule here. Outputs under outputs/sem2/r3/ (json and md committed, npz not).
- **Host cost, sized from R1's records** (e3v, 8 threads): MetaQA's fits ran 3,166 s for one fit, so three folds are about 3 hours; WebQSP's ran 12,642 s, and this fit has 70 percent of the relations; the reads are four scorers over about 160 test relations at 200 edges each on 2.59 million nodes. The rows are queued after the code passes its self-test.

### 11.12 Amendments found while writing the code (25 November 2026, 15:31 IST, 10:01 UTC), before any R3 number

Writing zsem2r3.py against zsem2.py showed that eight lines of 11.1-11.11 do not match what the code can do, or are ambiguous. None changes a call, a threshold or an expectation. No R3 number exists yet; the only numbers seen are the self-test's, on a planted toy (a graph of 10,800 nodes and 41 relations in 8 families of five, built to have the effect).

- **(a) S.rare is not "always seen" (11.3).** R1 pools the S relations under 500 training edges into one operator, S.rare. Those relations have no operator of their own, so they are **omitted** from this fit and this read: not trained on, not held out. The R labels with an operator of their own (at least 500 training edges) are train labels. The relations that can be held out are the S relations with an operator of their own, as 11.3 says first.
- **(b) What g outputs (11.4).** g gives D'/2 = 704 angles **and** D'/2 log plane weights. Directed relations use the angles (R1's rotation); undirected R labels use the plane weights (R1's symmetric part, 7.11 (b)): weight = exp(output), so the zero start gives weight 1 and every score the cosine. g has about 0.66 million weights (1,152 x 256 + 256 + 256 x 1,408 + 1,408), not 0.48 million; P adds 1,408 x 1,152.
- **(c) Equal share (11.5)** counts every train label, the R labels and the train S relations alike: share = 200,000 divided by the number of train labels, per epoch, up to what a label has.
- **(d) G-0 (11.9)** is read two ways: R1's check on the dev edges (largest score difference at most 1e-4, MRR difference at most 1e-3) **and** the equal-rank check on up to 300 evenly spaced test edges (at least 99 percent of the ranks of the name net at its start equal the cosine's).
- **(e) G-S, names (11.5):** compared on the **name strings**, not the label ids: if a dev or test relation's string equals a train relation's, the row stops (the two would have one vector). The names record counts distinct strings beside the labels.
- **(f) The mean name (11.6)** is the plain mean of the train relations' name vectors, **not divided by its norm** (as written: "the mean of the train relations' name vectors").
- **(g) Relations without a test edge** (none of its edges has a test source) are dropped from the calls and counted in the record. A relation counts once, whether it has 3 test edges or 200.
- **(h) The free-angle ceiling and the share it is recovered at (11.6, 11.7 "beside")** are **not in this code.** They need R1's stored model on the same edges, and WebQSP's is on the host only. They are a separate later command, named here, and nothing in a call depends on them.
- **(i) Fit length is declared (11.5): 10 epochs at most of up to 200,000 edges, learning rate 1e-3.** The record states how each fit ended (patience, or the last allowed epoch) and whether the best epoch was the last. **A fit that ends on the epoch limit with its best epoch last is read as under-fit, and a LEVEL there is a statement about the fit, not about the names;** a longer fit is then named, not run.
- **(j) Checkpoints.** A checkpoint after every epoch and the ranks of every chunk of 4,000 edges are stored, keyed to the code, the roles, the settings and zsem2.py; a restart takes up where it stopped, so a kill costs at most one epoch or one chunk. The checkpoint folder is the row's own temporary output and is removed when the row ends.
- **(k) A smoke run** (`fit --smoke`: one epoch on 4,096 edges, 20 test edges per relation) is a code check on the real data. It writes under outputs/sem2/r3/smoke/, which nothing reads; **its numbers are not read, not recorded and not used.** A change to the code after it is for a crash only.

### 11.13 Amendment from the MetaQA smoke row (G-0, test edges) (25 November 2026, 16:39 IST, 11:09 UTC), before any R3 number is read

The smoke row r3smoke-metaqa (fold 0: 9 relations, train 5, dev 1, test 3) stopped at G-0. The dev check passed (largest score difference 4.8e-7, MRR difference 8e-11). The test-edge check read 60 edges and **58 equal ranks, 0.9667, below the 0.99 of 11.12 (d)**. A debug run of the same fold's start on the host (the committed code with one print added, not committed, stopped right after the G-0 numbers) showed **2 of 60 ranks differ, each by exactly 0.5** (28,390.0 against 28,389.5 and 12,274.0 against 12,274.5).

**Cause.** The rank read counts a node whose score is within 1e-6 of the true tail's as a tie (half a position, zsem2.rank_read). The extra rows of P start at N(0, 1e-4^2), so the name net at its start differs from the cosine by up to about 5e-7, and one neighbour node moves in or out of that tie window. The scores are the same to 5e-7; only the tie bookkeeping moved by half a position. No number of the fit was read, and the smoke's numbers are not read (11.12 (k)).

**Amendment to 11.12 (d), the test-edge half of G-0.** The strict equality is replaced by two checks on the same up to 300 evenly spaced test edges:

- the largest difference between the name net's true-tail score at its start and the cosine's is **at most 1e-4** (the dev check's bound);
- **at least 99 percent of the ranks agree within max(1, 0.001 x the cosine's rank) positions.** The relative part is there because WebQSP has 2.59 million nodes, where a window of 1e-6 can hold more than one node; MetaQA's ranks differ by half a position.

The share of exactly equal ranks is still recorded, beside the two. Nothing else changes: not a call, a threshold, an expectation, the roles or the fit. The check reads the start of the net before any step, so it cannot move a result. The failed MetaQA smoke row is re-sent by the feeder under its own name with the amended code (it wrote nothing the report reads); the fit rows, and the WebQSP smoke and fit rows, had not run, so they run the amended code too. Every fit record carries the sha of the code that fitted it, and the report stops on a mismatch.

### 11.14 Records and call (25 November 2026, 21:03 IST, 15:33 UTC)

**Rows and gates.** r3names-metaqa and r3names-webqsp, then r3fit-metaqa0, 1, 2 and r3fit-webqsp0 (8 threads), all with outputs/mp_unified/zsem2r3.py at sha256 c75f3324fb6c8a4d, the code of 11.12-11.13; the report stops on a record fitted by other code, and none was. Every gate of every one of the four fits is PASS: G-N (the name encode reproduces 64 stored node vectors, cosine at least 0.999998 on every fit), G-E, G-S and G-0, G-0 by the amended rule of 11.13 (WebQSP: largest test-edge score difference 4.2e-7, ranks agreeing within the amended window on 100 percent, exactly equal on 73.3 percent). The first MetaQA smoke row stopped at that gate and is the one 11.13 describes; no number of it was read. The smoke rows are code checks and nothing here reads them.

**The fits.**

- **WebQSP (one fold):** 549 train relations, 78 dev relations, **163 test relations**, 28,051 test edges (at most 200 per relation), 20,789 dev edges. The fit ended by patience at epoch 5 with its best epoch 3 (dev MRR on the dev relations 13.62, from 3.0 at the start, which is the cosine), so it was not cut by the epoch limit and 11.12 (i) does not read it as under-fit. 9320 s.
- **MetaQA (three folds):** fold 0: 5 train relations, best epoch 6 of 8, best dev MRR 1.47; fold 1: 6 train relations, best epoch 3 of 5, best dev MRR 1.27; fold 2: 6 train relations, best epoch 0 of 2, best dev MRR 0.08. All three ended by patience, none at the limit. **Fold 2's best epoch is the start:** the net never beat the cosine on its dev relation, and its two test relations score the same under all four scorers.

**The calls (11.7; bootstrap over relations, 2,000 resamples, seed 0; MRR in points).**

| dataset | test relations | Delta = name - cosine | Gain from the name | call | d_r > 0 | mean MRR cosine / mean name / name / shuffled |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 7 | -0.12 [-0.54, +0.24] LEVEL | +0.24 [+0.04, +0.45] ABOVE | NO | 2 | 0.58 / 0.23 / 0.46 / 0.40 |
| webqsp | 163 | +11.98 [+8.99, +15.23] ABOVE | +6.09 [+3.78, +8.77] ABOVE | TRANSFERS | 126 | 2.63 / 8.51 / 14.60 / 9.66 |

- **WebQSP: Delta +11.98 [+8.99, +15.23] ABOVE; Gain from the name +6.09 [+3.78, +8.77] ABOVE: TRANSFERS.**
- **MetaQA: Delta -0.12 [-0.54, +0.24] LEVEL; Gain from the name +0.24 [+0.04, +0.45] ABOVE: NO.**
- **R3(c) verdict (11.7): PARTIAL (TRANSFERS on one of two).**

**Beside, deciding nothing (11.7).**

| | WebQSP (163 relations) | MetaQA (7 relations) |
|---|---|---|
| mean name minus cosine | +5.88 [+4.10, +7.74] ABOVE | -0.35 [-0.95, +0.14] LEVEL |
| name minus shuffled name | +4.94 [+2.51, +7.55] ABOVE | +0.06 [-0.05, +0.21] LEVEL |
| shuffled name minus cosine | +7.04 [+4.57, +10.08] ABOVE | -0.18 [-0.61, +0.24] LEVEL |
| relations with d_r above 0 | 126 of 163 | 2 of 7 |
| mean MRR, cosine / mean name / name / shuffled | 2.63 / 8.51 / 14.60 / 9.66 | 0.58 / 0.23 / 0.46 / 0.40 |

WebQSP pooled over its 28,051 test edges (each edge once, not each relation): **MRR cosine 2.52, mean name 8.02, shuffled 9.32, name net 14.53; hits@10 6.13 / 16.52 / 16.68 / 26.11; hits@100 19.08 / 36.68 / 34.17 / 44.09.** The free-angle ceiling and the share of it the name recovers are not read (11.12 (h): they need R1's stored WebQSP model, which is on the host).

### 11.15 Expectations against the outcome (11.8)

- **X18, WebQSP Delta ABOVE: held** (+11.98).
- **X19, WebQSP Gain from the name ABOVE: held** (+6.09).
- **X20, MetaQA Delta not ABOVE: held** (-0.12 LEVEL).
- **X21, shuffled name below the name net on WebQSP: held** (name minus shuffled +4.94 ABOVE).

All four held. They were written before any number, and the WebQSP hold-out was drawn by a hash of the relation's name.

### 11.16 What the numbers say, and what they do not (and 11.10, applied)

- **WebQSP.** A relation the fit never saw, given an operator from its name alone, ranks its true tails well above the cosine: pooled MRR 2.5 to 14.5, hits@100 19 to 44, and 126 of the 163 held-out relations improve. By the beside rows the gain has two parts: the net's relation-blind operator (the mean name) already gives +5.88 over the cosine, which is the shared map and a generic operator learned from 549 relations, and **the right name adds +6.09 on top**; a wrong name (shuffled) lands between (+7.04 over the cosine and 4.94 below the right name). The name matters; it is not all of the gain.
- **What the WebQSP mean hides.** The call counts each relation once, and the mean over relations (+11.98) is carried by a minority: the **median relation gains +1.5 points**, the largest gains are +99.0 (food nutrients), +74.5 (tv regular personal appearance person) and +70.6 (sports league participation team), and the five worst lose 5.1 to 14.7 (topic notable for -14.7). The interval is over relations and accounts for that spread; it is the right interval for the claim, but "most relations gain a little, a few gain a lot" is the shape.
- **Where it transfers.** The held-out relations belong to the same knowledge base as the train relations, and X18's reason was that sibling relations share words in their names (the award honor family). This is **transfer to new relations of a known graph, not to a new graph.** A net fit on WebQSP's relations and applied to MetaQA's is the cross-dataset read, named in 11.10, not declared and not run.
- **MetaQA: NO, and a statement about power.** Seven relations, an MRR below 2.2 for every relation under every scorer, the cosine's mean at 0.58, at most 6 train relations to learn a map from names to angles, and 2 of 7 relations above the cosine. The name's +0.24 over the mean name is read beside a mean name that is below the cosine (-0.35) and a shuffled name that is level with the name net (+0.06), so it is not a transfer. X20 said this would be so.
- **11.10, applied:** TRANSFERS on one dataset means an operator for a new relation of the same KB needs no fit, and SEM2's generalization claim is **read** on WebQSP, **not carried to a layout.** No pool, link or PL change follows from R3, and nothing is fit on a question. **Named, not declared and not queued:** (1) the cross-dataset read (WebQSP's net on MetaQA's relations); (2) the free-angle ceiling and the share of it the name recovers (a new module on the host, since zsem2r3.py's records pin its sha). The user's word decides whether either runs.

## 12. R4 declared: sets, not points (25 November 2026, 17:30 IST, 12:00 UTC), before any R4 number

### 12.1 Why, and what was seen

- **The user, 25 November (the idea):** put the walks themselves in the semantic space, "like a semantic tree or neuron where each vector entity is linked in different rotations or operations denoting the relation and branches out further and further". I answered that one point cannot stand for many nodes (a genre has many films, a film has many actors), which is why the literature uses boxes or distributions. **The user:** "yeah so we check the distribution across a specific domain which is the edge or operation and we would be able to walk all the nearest neighbours quickly instead of backtracking etc like we currently do / try testing this".
- **What R1 read, and what it did not.** R1 (section 7) moves a source to ONE point and ranks one true tail among all nodes (filtered rank, MRR). R2a and R2c (sections 9, 10) used that point to seed a one-hop. Neither read how much of a relation's WHOLE tail set the moved point recovers, and neither gave the operation a spread. That is the gap the user's idea names, and it is checkable on the stored operators with no new fit of the operators.
- **Prior art (from memory, not rechecked; recheck before any claim):** set and region embeddings for logical queries (GQE, Query2Box with boxes, BetaE with distributions, CQD), and typed ranges in knowledge-graph embeddings. This page claims no novelty.
- **Seen before this page:** R1's records (rotation MRR vs cosine, 7 of 7 LEARN on e3v), R2a's and R2c Part A's calls, R3(c)'s interim MetaQA read (a different question). **Not seen:** any recall of a tail set, by any scorer, on any dataset. No R4 number exists.

### 12.2 The question

For a source node s and one of R1's read labels L whose true tail set T(s, L) has m members (2 <= m <= 100): **does a scorer that gives the operation a region (the distribution of its tails around the moved source) put more of T in its top m than R1's point does?** Mean over units. Main measure: **R-precision**, the share of T in the top m. Beside it: recall at 4m (the share of T in the top 4m).

### 12.3 Units

- A unit is (a source s, a read label L). The read labels are R1's read groups (section 7.7): the R labels and the top S relations of each dataset, and the S relations pooled in S.other (a unit there is a source and ONE relation). T(s, L) is every true tail under L in the graph, **any split**; for an undirected R label it is every neighbour of s under that label. Sets of 2 to 100 are read; a set of one is R1's own read, a set over 100 is not a set a walk would take.
- **Sources** are test nodes (R1's node split, sha256 of "sem2|<dataset>|<node>"): up to **200 units per read group** (test) and up to **60 per group** (dev, on dev nodes), drawn with seeded generators (seed 0, streams 4/0 and 4/1). Each dataset's units are drawn from every eligible (source, label) of its groups; the eligible counts are recorded.
- **The source itself is the only node left out** of a unit's ranking. The other tails are NOT left out (they are what is retrieved). R1 left them out to read one tail; this read must not.
- Datasets: all seven, on **e3v** (the frozen arm is not run). R1's records are the operators.

### 12.4 The four scorers (every node y, with x = P v / |P v| and q = R1's operator applied to x_s; R1's stored P, rotations and plane weights, never refit)

| scorer | score of node y | what it is |
|---|---|---|
| cosine | v_s . v_y | no operator |
| point | q . x_y | R1's read |
| shift | x_y . (q + mu_op) | the point moved by the operation's mean residual (a type prior: where this operation's tails sit) |
| region | - sum_d W_d (x_y,d - q_d - mu_op,d)^2 | the diagonal Gaussian of the operation's residuals: mean mu_op, per-dimension variance var_op |

- **mu_op and var_op** are the mean and the per-dimension variance of x_y - q over the train-train edges of every label sharing the operator, up to 20,000 edges per read group (R1's operator index: the pooled rare relations share one). An operator with under 50 such edges gets mu = 0 and the pooled variance.
- **W** = 1 / ((1 - lam) var_op + lam mean_d(var_op)), scaled to mean 1 over d (a constant per operator: no unit's order changes). **lam in {1, 0.5, 0.2, 0.05}, chosen on the dev units by R-precision** (ties to the larger lam). lam = 1 is the isotropic case and ranks as the shift (an offset, no shape); smaller lam gives the dimensions their own spread. **region** is the chosen lam; **shift** is lam = 1 and is always read beside it, so a gain from the region is told apart from a gain from the offset.
- Nothing here is trained by gradient. The two fitted pieces (mu, var) are sample statistics of R1's train edges; the one chosen number (lam) is chosen on dev units before any test unit is scored.

### 12.5 The read

One streaming pass over all nodes per unit set: every scorer's score for every node, the top K = 400 kept per unit and scorer (exact: a block's candidates enter only if they beat the current K-th). The dev pass (the lam grid and the point) runs first and fixes lam; the test pass (cosine, point, shift, region) follows. R-precision uses the first m of the list, recall at 4m the first 4m. A checkpoint of the running lists is written every four minutes and the units are kept after they are built (a kill costs minutes).

### 12.6 The calls and the carry rule

- **Per dataset, main call:** region minus point, mean R-precision over the pooled test units (equal share per group by the cap, as in R1), paired bootstrap over units (2,000 resamples, seed 0, zsem2.paired): ABOVE (the 95% interval is above 0), LEVEL, BELOW. Beside it, same form: point - cosine, region - cosine, region - shift, shift - point (R-precision) and region - point, point - cosine, region - cosine (recall at 4m); region - point by set size (m 2-3, 4-10, 11-100) and per group (means).
- **Carry rule (fixed now): REGION_CARRIES if region - point is ABOVE on at least 4 of the 7 e3v datasets and BELOW on none; otherwise NOT_CARRIED.** The reads beside are named and decide nothing (a region that beats the point only because of the shift is told by region - shift, and said so).

### 12.7 Expectations, written before any number (a failed one is reported, not explained away)

- **point - cosine ABOVE on most datasets** (R1's 7 of 7 LEARN is on single tails; sets are a wider read of the same operators), smaller than R1's MRR gains because a set is ranked on its first m, not on one tail.
- **shift carries a part of any gain**, most on typed KB relations (MetaQA, WebQSP: the tails of one relation sit in one region of the space), least on the rule-link labels (R.*: the cosine neighbourhood is already the set).
- **region - shift is small.** A diagonal Gaussian over 1,408 frozen-encoder dimensions is a weak shape. **My expectation is NOT_CARRIED, with REGION (ABOVE) on at most two or three datasets, KB-structured first.** A BELOW on the rule-link groups is plausible (the hub line's null again: the operator adds nothing where the neighbourhood is the set).
- **By set size:** any gain rises with m (a larger set is where one point is least enough).
- If the region equals the shift, the gain is a type prior, not a distribution, and is reported as that.

### 12.8 What this does not test (named, not run)

- **The reverse direction** (genre to films): R1's directed operators were trained forward; the inverse relation is not an operator. Undirected R labels are read both ways.
- **Composition** (a region of regions, the walk's second and third hop, Query2Box-style projection of a box through a relation): the user's tree. It needs the one-hop set read to work first.
- **Whether a set read replaces an explicit walk** in pool coverage or retrieval: the stored links ARE the sets (R-precision 1 by definition), so the value of a region read is where a node's links are missing or noisy. That is a different read (links withheld from the walk) and is not declared here.
- This is SEM2's, not the journal's work.

### 12.9 Gates (a failure stops the row)

- **G-E** (the loader's own: the graph's family counts equal UL1's grade).
- **G-R** (zsem2t.operators): R1's test sample and ranks are reproduced from its stored operators (rank agreement >= 0.999, MRR within 0.01).
- **G-U:** unit sources are test (dev) nodes; every unit's set has 2 to 100 members; the sets read from the graph's keys equal the degree read (an independent count) for every unit.
- **G-T:** for 16 test units and every scorer, the stored top-K scores equal the definition's scores (up to the constant a query's score drops) within 5e-3, and no node outside the stored list, among the list's neighbours and 20,000 random nodes, scores above the K-th stored score.
- **G-0 (self-test):** a region with mu = 0 and equal weights returns the point's list.
- **Self-test** (planted toy: a one-to-many relation whose tails have a planted offset and anisotropic spread, a control with isotropic spread and no offset, an undirected clique relation): the streaming lists equal a full read for every scorer; the planted region beats the point and the shift; on the control it does not.

### 12.10 Code, rows and outputs

- **outputs/mp_unified/zsem2r4.py** (mine; imports zsem2, zsem2t.operators and zsem2h read-only and edits none): `selftest`, `run --ds D --host --threads 8` (per dataset), `report` (laptop).
- **Rows:** a code check on each loader first (r4smoke-squad on the six-dataset loader, r4smoke-multihoprag on MultiHop-RAG's; `--smoke`: 12 test and 6 dev units per group, written under outputs/sem2/r4/smoke/, read by nothing), then **r4-<dataset> for the seven**, 8 CPUs each, memory as R1's rows (MultiHop-RAG 4, SQuAD 6, MetaQA 6, MuSiQue 10, HotpotQA 32, 2Wiki 36, WebQSP 40 GB), the small ones first. The feeder retries a failed row once by its own name.
- **Outputs:** outputs/sem2/r4/<dataset>/e3v.json and .npz (per-unit R-precision and recall for every scorer), then `report` writes outputs/sem2/r4/report.json and report.md; the report stops on a record fitted by other code than the one it runs.

### 12.11 What follows (fixed now)

- **REGION_CARRIES:** the next page is the reverse direction and one composition step (region of region), on the datasets that carried; nothing is queued before the user is told.
- **NOT_CARRIED:** R4 closes this form (a diagonal Gaussian over R1's residuals does not beat the point as a set read). What it leaves open is named, not queued: a per-operation box trained by gradient on the sets (Query2Box's form), and the withheld-links read of 12.8. If the shift carries the gain, that is reported as a type prior, which R1's operators already approximate.
- Either way: no pool, link or PL change follows from R4, and no model is trained.

### 12.12 Records and call (25 November 2026, 20:42 IST, 15:12 UTC)

**Rows and gates.** The seven e3v rows ran (r4-multihoprag, r4-squad and r4-metaqa at 4 threads; r4-musique8, r4-webqsp, r4-hotpotqa and r4-2wiki at 8), all with outputs/mp_unified/zsem2r4.py at sha256 3a392d42afd4a699, and the report read them together at 15:07 UTC (20:37 IST); it stops on a record fitted by other code, and none did. **G-E, G-R, G-U and G-T PASS on all seven.** The first MuSiQue row ran at 4 threads and stopped at G-R (rank agreement 0.998667, below the 0.999 of 12.9): the float32 reduction order of R1's stored operators depends on the thread count, and R1 ran at 8. No number of that row was read; it was re-run at 8 threads under the name r4-musique8 and passed. Reading time per dataset: 4 s (MultiHop-RAG) to 1,620 s (WebQSP; 2Wiki 1,601, HotpotQA 1,227); measured peak memory WebQSP 15.8, HotpotQA 18.0, 2Wiki 20.5 GB (the feed asked 18, 22 and 24; 12.10's 40, 32 and 36 were upper bounds). The smoke rows are code checks and nothing here reads them.

**The calls (R-precision, share of the true tail set in the top m, mean over test units; paired bootstrap, 2,000 resamples, seed 0).**

| dataset | units | mean m | lam* | R-prec cosine | point | shift | region | region - point | region - shift | point - cosine | call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 46 | 62.72 | 1.0 | 34.85 | 41.92 | 41.75 | 41.75 | -0.17 [-0.36, +0.01] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +7.07 [+4.55, +9.28] ABOVE | LEVEL |
| squad | 448 | 35.81 | 1.0 | 36.92 | 48.61 | 48.3 | 48.3 | -0.31 [-0.76, +0.12] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +11.69 [+9.93, +13.40] ABOVE | LEVEL |
| musique | 600 | 14.92 | 0.5 | 11.06 | 22.19 | 19.82 | 18.52 | -3.66 [-4.79, -2.59] BELOW | -1.30 [-2.19, -0.46] BELOW | +11.12 [+9.21, +12.95] ABOVE | BELOW |
| metaqa | 1433 | 4.57 | 1.0 | 5.74 | 16.68 | 15.65 | 15.65 | -1.02 [-1.71, -0.30] BELOW | +0.00 [+0.00, +0.00] LEVEL | +10.94 [+9.82, +12.08] ABOVE | BELOW |
| webqsp | 2639 | 5.93 | 1.0 | 7.84 | 38.51 | 37.9 | 37.9 | -0.61 [-1.18, -0.01] BELOW | +0.00 [+0.00, +0.00] LEVEL | +30.67 [+29.29, +32.10] ABOVE | BELOW |
| hotpotqa | 800 | 8.02 | 1.0 | 12.59 | 17.82 | 16.02 | 16.02 | -1.80 [-2.41, -1.16] BELOW | +0.00 [+0.00, +0.00] LEVEL | +5.23 [+4.21, +6.25] ABOVE | BELOW |
| 2wiki | 800 | 11.03 | 1.0 | 8.95 | 14.98 | 12.71 | 12.71 | -2.27 [-2.90, -1.66] BELOW | +0.00 [+0.00, +0.00] LEVEL | +6.03 [+5.13, +6.92] ABOVE | BELOW |

**Region minus point: ABOVE on 0 of 7, BELOW on 5 (MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki), LEVEL on 2 (MultiHop-RAG, SQuAD). Rule 12.6 (at least 4 of 7 ABOVE and none BELOW): NOT_CARRIED.** On the dev units no region beat the point either, on all seven (the dev grid: the point's R-precision is at or above every lam; for example MuSiQue 21.32 against 19.45 at lam 0.5, 2Wiki 16.51 against 13.89 at lam 1), so the choice of lam did not hide a region that the test units then lost.

**By set size (region minus point, R-precision).** ABOVE in none of the 19 size bins; BELOW in 11, LEVEL in 8.

| dataset | m 2-3 | m 4-10 | m 11-100 |
|---|---|---|---|
| multihoprag | - | - | 46: -0.17 [-0.36, +0.01] LEVEL |
| squad | 28: -4.17 [-8.93, +0.00] LEVEL | 45: -1.12 [-3.66, +1.18] LEVEL | 375: +0.07 [-0.20, +0.34] LEVEL |
| musique | 168: -3.17 [-5.65, -0.89] BELOW | 188: -3.46 [-6.04, -1.11] BELOW | 244: -4.16 [-5.46, -2.90] BELOW |
| metaqa | 956: +0.24 [-0.75, +1.26] LEVEL | 339: -2.23 [-3.26, -1.29] BELOW | 138: -6.86 [-8.29, -5.61] BELOW |
| webqsp | 1489: -0.41 [-1.37, +0.49] LEVEL | 827: -0.33 [-1.30, +0.61] LEVEL | 323: -2.20 [-3.13, -1.35] BELOW |
| hotpotqa | 276: -2.29 [-3.68, -0.79] BELOW | 335: -2.36 [-3.25, -1.52] BELOW | 189: -0.06 [-0.55, +0.43] LEVEL |
| 2wiki | 220: -2.73 [-4.32, -1.29] BELOW | 300: -2.96 [-4.06, -1.93] BELOW | 280: -1.16 [-1.73, -0.58] BELOW |

**Recall at 4m, beside (region minus point: BELOW on SQuAD, MuSiQue, HotpotQA, 2Wiki; LEVEL on MultiHop-RAG, MetaQA, WebQSP).**

| dataset | region - point | region - cosine |
|---|---|---|
| multihoprag | -0.15 [-0.55, +0.25] LEVEL | +4.72 [+2.11, +7.34] ABOVE |
| squad | -0.51 [-0.92, -0.11] BELOW | +10.94 [+9.06, +12.76] ABOVE |
| musique | -3.91 [-5.25, -2.65] BELOW | +14.26 [+12.10, +16.38] ABOVE |
| metaqa | +0.61 [-0.07, +1.34] LEVEL | +16.88 [+15.22, +18.68] ABOVE |
| webqsp | +0.61 [-0.08, +1.31] LEVEL | +34.63 [+33.05, +36.12] ABOVE |
| hotpotqa | -3.72 [-4.67, -2.86] BELOW | +7.78 [+6.66, +8.97] ABOVE |
| 2wiki | -4.25 [-5.12, -3.42] BELOW | +6.61 [+5.59, +7.56] ABOVE |

### 12.13 Expectations against the outcome (12.7; a failed one is reported, not explained away)

- **Point minus cosine ABOVE on most datasets: held, 7 of 7** (+5.23 on HotpotQA to +30.67 on WebQSP; at recall 4m the region minus cosine is ABOVE on 7 of 7 too, +4.72 to +34.63). R1's operators, learned on single tails, already put more of a whole tail set in the top m than the cosine does, on every dataset. (The second half of that expectation, that the set gain is smaller than R1's MRR gain, was not tabulated and is not claimed.)
- **Shift carries a part of any gain, most on typed KB relations (MetaQA, WebQSP): not held.** Shift minus point is BELOW on five datasets (MuSiQue -2.36, MetaQA -1.02, WebQSP -0.61, HotpotQA -1.80, 2Wiki -2.27) and LEVEL on two (-0.17, -0.31); it is ABOVE on none, MetaQA and WebQSP included. **A description of the per-group means in the records (no intervals, not a call, found after the numbers were seen):** the offset is above the point where the relation's tails are a small fixed vocabulary (MetaQA has_genre 38.29 to 44.96, in_language 29.51 to 32.79, release_year 3.37 to 3.97; WebQSP person profession 9.46 to 18.50, film genre 20.60 to 22.39, S.other 15.10 to 20.75) and below it where the tail set depends on the source (MetaQA R.name_comention 63.84 to 52.28, WebQSP film starring 94.16 to 86.80, MuSiQue R.mention 34.84 to 30.08, HotpotQA hyperlink 18.05 to 11.94, 2Wiki hyperlink 16.92 to 11.37). That is a type prior, as 12.7's last bullet says it would be reported, and it nets below the point because the source-dependent groups lose more than the vocabulary groups gain.
- **Region minus shift is small: held.** It is 0.00 on six datasets, because lam = 1 won on dev there (the region IS the shift, not a distribution), and -1.30 BELOW on MuSiQue, the one dataset where a smaller lam (0.5) won on dev and then lost on the test units. A diagonal Gaussian over R1's residuals gave no shape that the isotropic offset does not.
- **My expectation, NOT_CARRIED with REGION above on at most two or three datasets: held (0 ABOVE).** "A BELOW on the rule-link groups is plausible": it came, on MuSiQue (whose groups are all R labels: R.mention 30.08 to 26.25 from shift to region, R.title_comention 10.58 to 9.82; R.name_comention went the other way, 18.81 to 19.49), and against the point: under the shift R.mention is below the point in five of its six datasets (SQuAD is +0.34 above).
- **By set size, any gain rises with m: not seen.** Region minus point is above 0 in no bin; it is at its lowest at large m on MetaQA (-6.86, 138 units at m 11-100) and flat or negative elsewhere. The one-to-many case the idea started from (a genre with many films) is where this read is weakest, not strongest.
- **If the region equals the shift, the gain is a type prior, not a distribution: that is the finding on six of seven,** and the type prior is below the point at the dataset level.

### 12.14 What this closes and what it leaves (12.11, applied; nothing is queued by this page)

- **R4 closes this form.** A diagonal Gaussian built from R1's train-edge residuals is not a better set read than R1's point on any dataset read. The point itself is a real set read (7 of 7 above the cosine), but the absolute levels are low (R-precision 15 to 49, MetaQA 16.7, 2Wiki 15.0), against 100 for the stored links, so the stored links stay the way sets are retrieved; the value of a semantic set read is only where links are missing or noisy (12.8), which is a different read and is not declared.
- **Limits.** The region here is the weakest shape (sample mean and per-dimension variance, no gradient, 1,408 dimensions); a box or distribution trained on the sets themselves (Query2Box's form) was not tested, and this page is not a result against it. It is named, not queued.
- **No pool, link or PL change follows, no model is trained.** The depth of the user's tree (a set read of a set read) was not read by R4; section 13 (R5, branching), declared before this filing, reads it. R5's rows are queued behind the UL2 chain.

## 13. R5 declared: branching, the second hop through R1's operators (25 November 2026, 20:20 IST, 14:50 UTC), before any R5 number

### 13.1 Why, and what was seen

- **The user, 25 November:** put the walks themselves in the semantic space as "a semantic tree or neuron where each vector entity is linked in different rotations or operations denoting the relation and branches out further and further". R4 (section 12) tested the first part of it, a set at ONE step. Later the same day: "what about the new type of semantic branching i told you to test?" and "if we are sure it doesnt improve do not waste time".
- **What R4 settled (six of seven datasets read; 2Wiki's row was still running; the carry rule of 12.6 is already failed by the four BELOW, so its verdict cannot change):** region minus point is BELOW on MuSiQue (-3.66 R-precision points), HotpotQA (-1.80), MetaQA (-1.02) and WebQSP (-0.61), LEVEL on MultiHop-RAG and SQuAD, ABOVE on none; the region equals the shift on four of the six. **The distribution is closed: no more time goes to a region, a box or a shape over R1's residuals.**
- **What R4 also showed, and R5 builds on:** R1's point, used as a ranker of a relation's WHOLE tail set (the moved source's nearest neighbours), is ABOVE cosine on all six read, by +5.2 to +30.7 R-precision points. That is one step of the tree working with no stored link. The depth ("branches out further and further") is what R4 did not read.
- **A correction to 12.11.** Section 12.11 tied composition to the region carrying ("the next page is the reverse direction and one composition step"). That tied the tree to the region, and the tree does not need it: a branch is a node the first step lands on, and the next step starts from that node. R5 is declared on R1's point, whose one-hop win R4 recorded.
- **Prior art (from memory, not rechecked; recheck before any claim):** path queries by projection in embedding space (GQE, Query2Box, BetaE, CQD's beam over the first hop), multi-hop retrieval by chained dense lookups. This page claims no novelty.
- **Seen before this page:** R4's six records (one hop only), R1's records, R2a and R2c's one-hop traversals. **Not seen:** any second-hop number, by any chain of operators or of cosines. No R5 number exists.

### 13.2 The question

From a test source s, apply operator L1 to s, keep the **b = 10** nodes it ranks highest (the branches), apply operator L2 from each branch, and rank every node by its best branch. **Does this walk reach the second-hop set T2 of (s, L1, L2) better than** (a) the same walk with cosine at both steps, (b) one composed point with no branching, (c) cosine to s alone? Main measure: R-precision, the share of T2 in the top m = |T2|; beside it, recall at 4m.

### 13.3 Units (no parameter is tuned, so there is no dev pass)

- **Chain labels:** the labels that have their own operator in R1's plan: every R label with an operator and every S relation with its own operator (a relation pooled into S.rare is not a chain label). T(x, L) is every neighbour of x under L (any split; both ends of an undirected R pair; the destination of a directed S edge).
- **A unit is (s, L1, L2):** s a **test** node (R1's node split); T1 = T(s, L1) with 1 <= |T1| <= 100; **T2 = (union over a in T1 of T(a, L2)) minus ({s} union T1)**, with **2 <= |T2| <= 100**. T2 is the set of nodes the second hop reaches that the first hop did not already give.
- **Draw (seeded, seed 0, streams 5/0 and 5/1):** a pool of up to 3,000 test sources that have an edge under some chain label; every (L1, L2) the pool realises is a candidate unit; the path group of a unit is (name of L1, name of L2); **groups with at least 20 candidate units are kept, the 12 largest by candidate count, up to 100 units per group.** A dataset with fewer than 20 test units in all is **not a dataset for this read** (listed, not counted). The eligible counts are recorded.
- Datasets: all seven, on **e3v**; R1's records are the operators (never refit); the frozen arm is not run.

### 13.4 The four scorers (x = P v / |P v|; q(x, L) = R1's operator L applied to x, as in R4's point)

| scorer | score of node y | what it is |
|---|---|---|
| cosine | v_s . v_y | no structure at all |
| chain-cos | max over branches a in B_cos of [ v_s . v_a + v_a . v_y ] | the same walk with cosine at both steps (B_cos: the b nodes with the highest v_s . v_a, s left out) |
| chain-point | max over branches a in B_pt of [ q(x_s, L1) . x_a + q(x_a, L2) . x_y ] | R1's operators, branching (B_pt: the b nodes with the highest q(x_s, L1) . x_a, s left out) |
| compose | q( q(x_s, L1) / |q(x_s, L1)| , L2 ) . x_y | the same two operators on ONE point, no branching |

- **Left out of every ranking:** s, every node of T1 (the first hop's own answers do not count as second-hop answers; the same for every scorer), and, for a chain scorer, y = a on the row of branch a (a node is not its own neighbour). Other branches are not left out.
- b = 10 is fixed here. No other value is read (the first hop's T1 can hold up to 100 nodes: the branch recall is capped by b, and is reported, not repaired).
- The score of a chain is a sum of two first-order scores of the same kind (cosines, or R1's dot products of unit vectors); nothing is weighted or learned.

### 13.5 The read

Pass 1: one streaming pass over all nodes per unit set gives the b branches and their first-hop scores for chain-cos and chain-point (R4's streaming read, K = b). Pass 2: one streaming pass over all nodes gives the top K = 400 of every scorer (exact, as in 12.5). Checkpoints: the units after they are built, the running lists every four minutes (a kill costs minutes). Measures per unit: R-precision, recall at 4m, and (beside) the **branch recall**: the share of T1 inside the b branches (min(|T1|, b) is the most it can be), and whether any branch is a true first-hop node.

### 13.6 The calls and the carry rule (fixed now)

- **Per dataset, main call:** chain-point minus chain-cos, mean R-precision over the pooled test units (equal share per path group by the cap), paired bootstrap over units (2,000 resamples, seed 0, zsem2.paired): ABOVE (the 95% interval above 0), LEVEL, BELOW.
- **Carry rule: BRANCH_CARRIES if chain-point minus chain-cos is ABOVE on at least 4 of the datasets read and BELOW on none; otherwise NOT_CARRIED.** At least 4 datasets must be read, or the call is INCOMPLETE.
- **Read beside (named, decide nothing):** chain-point minus compose (does branching matter), chain-point minus cosine and chain-cos minus cosine (does a walk beat ranking by cosine to s), the same on recall at 4m, by path kind (R then R, S then S, mixed), by |T2| (2-3, 4-10, 11-100), by |T1| (up to 10, over 10), per path group, and the branch recall.

### 13.7 Expectations, written before any number (a failed one is reported, not explained away)

- **Absolute levels fall.** R4's one-hop point reached 16.7 to 48.6 R-precision points on the six; a second hop needs a correct branch first, and b = 10 caps the branches. I expect chain-point R-precision in single digits to the low twenties, below the one-hop levels.
- **chain-point minus chain-cos:** ABOVE where R1's operators beat cosine most at one hop (WebQSP, the typed KB relations; the hyperlink relation of 2Wiki and HotpotQA), LEVEL or BELOW on R-then-R paths (the cosine neighbourhood already is the set). **My expectation is NOT_CARRIED, with ABOVE on at most two or three datasets.**
- **chain-point minus compose:** ABOVE on most (branching lands on real nodes and corrects the first step; one composed point carries both steps' error), smallest where R1's rotation is nearly exact.
- **cosine to s alone** may beat both chains on the text datasets (the second hop is often near s in a text encoder's space); where it does, the tree adds nothing over cosine there, and that is said.
- A dataset with almost no chain (MetaQA's relations all start at a movie) may be "not a dataset for this read"; that is a finding about the graph, not a failure.

### 13.8 What this does not test (named, not run)

- **The reverse direction** and any path through an inverse relation (MetaQA's real two-hop questions go movie, actor, movie): R1's directed operators were trained forward; a rotation's inverse is its transpose, but no read of it exists yet.
- **Three hops** and any width other than 10.
- **Whether a chain replaces an explicit walk in a pool or in retrieval** (the stored links ARE the sets, R-precision 1 by definition; the value of a semantic branch is where links are missing or noisy). That needs a withheld-links read and is not declared here.
- This is SEM2's, not the journal's work.

### 13.9 Gates (a failure stops the row)

- **G-E**, **G-R** as in 12.9 (R1's operators reproduced; `--threads 8`).
- **G-U:** unit sources are test nodes; |T1| and |T2| are in range; T2 contains neither s nor any node of T1; for 50 units T1 and T2 recomputed by a direct scan of the graph's keys (independent of the enumeration) equal the stored sets.
- **G-T:** for 12 units and every scorer, the stored top-K scores equal the definition's scores (to 5e-3) and no node outside the stored list, among the list's neighbours and 20,000 random nodes, scores above the K-th stored score (the chain scorers recomputed from the definition over the stored branches).
- **Self-test** (planted toy: a forward chain with noise at each step, a relation with a planted second hop, an undirected clique relation): the streaming lists equal a full read for every scorer; the exclusions hold; the planted chain beats the cosine chain and the cosine; branching beats the composed point where the first step is noisy.

### 13.10 Code, rows and outputs

- **outputs/mp_unified/zsem2r5.py** (mine; imports zsem2, zsem2t.operators and zsem2r4's streaming read, read-only; edits none): `selftest`, `run --ds D --host --threads 8` (per dataset), `report` (laptop). Run records carry script_sha256 and `report` stops on a mismatch.
- **Rows:** `r5-<dataset>` for the seven, 8 CPUs, memory sized from R4's measured peaks (WebQSP 15.8 GB, HotpotQA 18.0 GB); the small datasets first; `--threads 8`. The feeder retries a failed row once by its own name.
- **Outputs:** outputs/sem2/r5/<dataset>/e3v.json and .npz, then `report` writes outputs/sem2/r5/report.json and report.md.

### 13.11 What follows (fixed now)

- **BRANCH_CARRIES:** the next pages are the reverse direction and the withheld-links read of 13.8, on the datasets that carried; nothing is queued before the user is told.
- **NOT_CARRIED:** R5 closes the chained form on R1's operators; what it leaves open is named, not queued (the reverse direction, a third hop, a trained path scorer). The reads beside say whether any walk beats cosine anywhere.
- Either way: no pool, link or PL change follows from R5, and no model is trained.

### 13.12 Records and call (25 November 2026, 23:55 IST, 18:25 UTC)

Read by `zsem2r5.py report` from `outputs/sem2/r5/<dataset>/e3v.json` (every record carries the script's sha). **Six datasets read; MultiHop-RAG has no test unit under the chain labels (0 units, fewer than 20), so it is not a dataset for this read.** R-precision in points, mean over the pooled test units; the interval is the paired bootstrap over units (2,000 resamples, seed 0).

| dataset | units | cosine | chain-cos | chain-point | compose | chain-point minus chain-cos | call |
|---|---:|---:|---:|---:|---:|---|---|
| SQuAD | 710 | 15.70 | 17.25 | 24.92 | 20.44 | +7.68 [+6.40, +9.02] | ABOVE |
| MuSiQue | 900 | 6.43 | 6.11 | 9.25 | 7.66 | +3.14 [+2.32, +3.94] | ABOVE |
| MetaQA | 1,200 | 2.03 | 1.34 | 3.29 | 2.56 | +1.94 [+1.43, +2.50] | ABOVE |
| WebQSP | 1,200 | 1.50 | 1.40 | 2.52 | 3.07 | +1.12 [+0.43, +1.85] | ABOVE |
| HotpotQA | 1,200 | 2.93 | 2.88 | 4.22 | 3.65 | +1.34 [+0.98, +1.66] | ABOVE |
| 2Wiki | 1,200 | 2.10 | 1.90 | 3.58 | 3.03 | +1.68 [+1.40, +1.98] | ABOVE |

**Rule 13.6: chain-point minus chain-cos is ABOVE on 6 of 6 read and BELOW on none, at least 4 read: BRANCH_CARRIES.** Recall at 4m agrees on all six (chain-point minus chain-cos: SQuAD +10.19, MuSiQue +6.22, MetaQA +6.38, WebQSP +4.51, HotpotQA +3.22, 2Wiki +3.60 points, every interval above 0).

**Size, said plainly.** On four of the six the ABOVE sits near the floor of the scale: chain-point reaches only 2.5 to 4.2 R-precision points (MetaQA 3.29, WebQSP 2.52, HotpotQA 4.22, 2Wiki 3.58), a gap of +1.1 to +1.9 points over chain-cos, so chain-point is 1.5 to 2.5 times chain-cos's level, from a very low one. MuSiQue (9.25) and SQuAD (24.92) are the only two above 5 points. R-precision here is measured against sets that are the stored links themselves, so it says whether the operators reproduce a second hop, not that a pool or a retriever gains.

### 13.13 The reads beside (decide nothing)

- **chain-point minus compose** (does branching matter): ABOVE on SQuAD +4.48, MuSiQue +1.59, MetaQA +0.72, HotpotQA +0.57, 2Wiki +0.55; **BELOW on WebQSP, -0.54 [-1.08, -0.03]** (the upper end of the interval is at the edge).
- **chain-point minus cosine** (does the walk beat ranking by cosine to s): ABOVE on all six (SQuAD +9.23, MuSiQue +2.82, MetaQA +1.26, WebQSP +1.02, HotpotQA +1.29, 2Wiki +1.48). **chain-cos minus cosine:** ABOVE on SQuAD (+1.55), BELOW on MetaQA (-0.69 [-0.96, -0.43]), LEVEL on the other four. A walk with cosine at both steps is no better than cosine alone on four of the six (ABOVE only on SQuAD, BELOW on MetaQA); the gain comes from R1's operators.
- **The branches** (first hop, b = 10): chain-point's branches hold more of the true first-hop set than chain-cos's on five datasets (branch recall SQuAD 63.4 vs 53.7, MuSiQue 35.0 vs 23.7, HotpotQA 28.6 vs 20.0, 2Wiki 25.9 vs 16.5, MetaQA 2.5 vs 1.0) and equal on WebQSP (30.0 vs 30.3). On MetaQA neither scorer finds the true first hop (any true first hop 11.3% for chain-point, 3.6% for chain-cos), so its second hop is read at a very low level.
- **By path kind:** R then R is ABOVE on five (SQuAD, MuSiQue, MetaQA +1.09, HotpotQA +0.88, 2Wiki +1.19) and LEVEL on WebQSP (+0.06 [-0.33, +0.53]); S then S (100 units each, HotpotQA and 2Wiki only) ABOVE, +2.79 and +4.11; mixed ABOVE on all four that have it (MetaQA +2.12, WebQSP +1.22, HotpotQA +1.59, 2Wiki +1.79).
- **By set size:** the gain is largest where the second-hop set is large (|T2| 11 to 100: ABOVE on all six, +1.21 to +7.25) and mostly LEVEL where it is 2 to 3 (LEVEL on MuSiQue, WebQSP, HotpotQA, 2Wiki; ABOVE on SQuAD +9.78 and MetaQA +2.93). **By |T1|:** ABOVE on every dataset in both bins except WebQSP over 10 (+0.89 [-0.20, +1.93], LEVEL).

### 13.14 Expectations against the outcome (13.7; a failed one is reported, not explained away)

- *Absolute levels fall below the one-hop levels:* **HELD** (chain-point 2.5 to 24.9 points against R4's one-hop 16.7 to 48.6).
- *NOT_CARRIED, with ABOVE on at most two or three datasets:* **FAILED.** I expected ABOVE on at most two or three datasets; it is ABOVE on all six read, by 1.1 to 7.7 points.
- *chain-point minus compose ABOVE on most:* **HELD** (five of six; BELOW on WebQSP, at the edge of the interval).
- *Cosine alone may beat both chains on the text datasets:* **FAILED.** chain-point beats cosine on all six, text datasets included.
- *LEVEL or BELOW on R-then-R paths:* **FAILED.** ABOVE on five of six; LEVEL only on WebQSP.
- *A dataset with almost no chain may not be read:* MetaQA was read (1,200 units); the dataset not read is MultiHop-RAG, which has 0 test units under the chain labels.

### 13.15 What this closes and what it leaves (13.11, applied; nothing is queued by this page)

- **Closes:** the question of 13.2 on R1's operators: a two-step branched walk through them reaches the second-hop set better than the same walk by cosine, than one composed point (five of six) and than cosine to s, on all six datasets read, by small absolute amounts on four of them.
- **Leaves, named and not run:** (a) **the withheld-links read of 13.8**, the one that says whether a semantic branch is worth anything where links are missing or noisy, which is the only place a stored set does not already give R-precision 1; (b) the reverse direction (MetaQA's movie, actor, movie); (c) a third hop and other widths; (d) a trained path scorer. By 13.11 nothing is queued before the user is told. No pool, link or PL change follows, no model is trained, and this is SEM2's work, not the journal's.

## 14. R6 declared: do link operators cross between passages and a KB? (25 November 2026, 22:05 IST, 16:35 UTC), before any R6 number

### 14.1 Why, and what was seen

- **The user, 25 November:** "we also need to test passage to KB and vice versa that is the major issue, a simple common semantic space will not solve it hence we are doing this operators etc". Asked whether that meant link level (operators on links) or question level (an MLP or GNN trained on passage datasets and scored on KB datasets), **the user:** "if we can transfer links only then we can work with them during questions with mlp and gnn so both are technically linked". **This page is the link-level read. The question-level read is not declared: it follows only if links cross.**
- **The two kinds of corpus.** Passage corpora (MultiHop-RAG, SQuAD, MuSiQue, HotpotQA, 2Wiki) have text nodes and link labels built from text (the R labels: mention, name comention, title comention) and, on HotpotQA and 2Wiki, a structural S relation (hyperlink). KB corpora (MetaQA, WebQSP) have entity nodes whose text is a name and facts, many typed S relations (7 on MetaQA, 791 operators on WebQSP) and the same R labels (two on MetaQA, three on WebQSP).
- **What the common space gives, and what is on file.** The common semantic space is the cosine of the frozen e3v vectors: the baseline of every section of this page's family (R3: WebQSP's cosine MRR 2.6 on held-out relations, MetaQA's 0.6). R3(c) (section 11) fitted a name net g(name vector) -> angles and plane weights on **one** KB's relations and read it on that KB's held-out relations (WebQSP TRANSFERS, MetaQA NO). It never left the dataset it was fitted on, and no operator has ever been read across corpora or across types.
- **A correction.** My handoff note of 25 November 21:12 said R3's fitted weights were not stored. They are: every fold's npz holds the best-epoch state (model.P and model.g.*), and outputs/sem2/r3/webqsp/f0.npz is on the laptop. **The KB net below is that stored net, not a refit.**
- **Seen before this page:** R1's, R3's and R4's records. **Not seen:** any score of any operator on a corpus other than the one it was fitted on. No R6 number exists. This page claims no novelty (prior art on cross-domain transfer of relation embeddings is not rechecked here).

### 14.2 The question

**Does an operator net fitted on the links of one type of corpus rank the true tails of the other type's links above plain cosine (the common space), and above a relation-blind operator the same net produces?** Four directions, each read on e3v with no refit on the target:

1. **KB -> KB** (control and the named cross-dataset read): the WebQSP net on MetaQA's S relations.
2. **KB -> passage:** the WebQSP net on the link labels of the five passage corpora.
3. **passage -> KB:** a net fitted on HotpotQA's links, on WebQSP's held-out relations and MetaQA's S relations.
4. **passage -> passage** (control): the HotpotQA net on the other four passage corpora's labels.

### 14.3 The nets (both e3v, D = 1,152, D' = 1,408, hidden 256, as R3; code of section 11 for the net, the loss and the optimiser)

- **N_KB:** R3's WebQSP fit as stored (best epoch 3 of 5; trained on 549 train relations: the three R labels and 546 S relations; stopped on 78 dev relations; 163 relations held out). Its stored file and its sha are pinned by R3's record; the row stops on a mismatch. **Names seen by N_KB:** the 549 train names, **including mention, name comention and title comention.**
- **N_P:** a new fit on **HotpotQA's four labels with an operator of their own** (R.mention, R.name_comention, R.title_comention, S.hyperlink), all of them train. There is no relation left to hold out, so the stop is on **dev edges of these four labels** (dev-node sources), as R1's fit (7.5); two epochs without a gain, 10 at most, the best epoch kept; an epoch is up to 200,000 train edges with an equal share per label; learning rate 1e-3, temperature 0.05, seed 0. The last layer of g starts at zero, so at the start every score is the cosine (G-0, as R3). **Names seen by N_P:** the four. HotpotQA is the passage corpus because it carries all four passage-side labels, the structural one included. 2Wiki carries them too and is read, not fitted. A second passage net is named below, not run.
- **No target corpus is ever fitted on.** A target's labels get their operators from the net and their own name string alone.
- **Name vectors:** as 11.4 (the e3v encoder on the name string, divided by its norm; G-N on every names file). The names of the three R labels come from R3's names files; "hyperlink" is encoded by R3's names command on HotpotQA (the same string on 2Wiki).

### 14.4 Units and the read

- **A unit is one (target corpus, label with an operator of its own).** Test sources only (R1's node split); up to **200 test edges per unit** (R3's sampler, default_rng(0)); the filtered rank of the true tail among all of the target's nodes (7.6), MRR, hits@10, hits@100. A unit with fewer than 20 test edges is dropped from the calls and counted.
- **Scorers on the same edges (as 11.6):** **cosine**; **name** (the net's operator from the label's own name); **mean name** (the same net on the mean of the net's train name vectors: a relation-blind operator the net produced); **shuffled name** (each target label takes another target label's name, a derangement, seed 0; a target with one label has none and the shuffled read equals the name read, said in the record).
- **Target units by direction:**
  - **MetaQA:** its 7 S relations with an operator of their own (the unit set of the main calls); its R.mention and R.name_comention are read and reported beside.
  - **WebQSP:** its **163 held-out relations of R3** (u < 0.20 of 11.5), the same relations R3 read, for N_P (for N_KB they are R3's own read, not re-read); its three R labels beside.
  - **Passage corpora:** every label with an operator of its own: HotpotQA 4, 2Wiki 4, MuSiQue 3, SQuAD 3, MultiHop-RAG 1 (R.name_comention).
- **Rows:** N_KB on each of the seven datasets (WebQSP: the loader gate only); N_P on each of the seven; HotpotQA's own four labels under N_P are an in-sample read (new nodes, fitted labels), beside.

### 14.5 The calls (11.7's form, per direction)

- **d_u = MRR(name) - MRR(cosine)** on unit u's edges; **e_u = MRR(name) - MRR(mean name).** Each unit counts once; the interval is a bootstrap over units (2,000 resamples, seed 0, 95%); ABOVE, LEVEL or BELOW as 9.6.
- **Per direction:** TRANSFERS if Delta (the mean of d_u) is ABOVE and the gain from the name (the mean of e_u) is ABOVE; **GENERIC** if Delta is ABOVE and the gain is not (a relation-blind operator carries it); **LOSES** if Delta is BELOW; **NO** otherwise.
- **The units pooled per direction:** (1) MetaQA's 7 S relations; (2) all 15 passage units; (3) the 163 WebQSP relations and MetaQA's 7 (170 units, WebQSP dominates; each dataset is also called alone); (4) the 11 units of 2Wiki, MuSiQue, SQuAD and MultiHop-RAG.
- **Bridge verdict (fixed now): BRIDGES if directions 2 and 3 both have Delta ABOVE; ONE_WAY (named) if exactly one has; NOT_BRIDGED if neither has; INCOMPLETE while a row is missing.** The direction 4 control is read first: **if direction 4's Delta is not ABOVE, operators do not even cross between corpora of one type, and the cross-type reads are reported as that, not as a result about types.**
- **Beside, deciding nothing:** the S.hyperlink units alone (the only passage-side name N_KB has not seen) against the R-label units (names it has seen); the mean name and the shuffled name against the cosine; relations (units) with d_u above 0; hits@10 and hits@100; MetaQA's and WebQSP's R labels under N_P; N_P's in-sample HotpotQA read.

### 14.6 Expectations, written before any number (a failed one is reported, not explained away)

- **X22 (KB -> KB, MetaQA):** Delta not ABOVE. MetaQA is at the floor (R3: MRR below 2.2 for every relation) and has 7 units.
- **X23 (KB -> passage):** Delta not ABOVE on the 15 pooled units, and BELOW or LEVEL on both S.hyperlink units. The net learned its R-label operators from KB text (a name and facts), and a structural link of a passage corpus is a kind of link it has no name for.
- **X24 (passage -> KB):** Delta not ABOVE. Four training names cannot teach a map from "directed by" or "film starring" to angles; the net's value at a new name is an interpolation between four points, and the one directed point is hyperlink. If anything is above it is the mean name, through the shared map, which would read GENERIC.
- **X25 (passage -> passage, the control):** Delta ABOVE (the R labels of one corpus type carry their operators across corpora: the same kind of text, the same link rules). If it fails, 14.5's rule applies.
- **X26 (N_P in-sample on HotpotQA, beside):** Delta ABOVE (the labels were fitted there).
- **My expectation for the verdict is NOT_BRIDGED.** Writing it first means a BRIDGES or ONE_WAY is a surprise, and the user said the opposite is the working idea.

### 14.7 Gates (a failure stops the row)

- **G-E** (the loader's own), **G-N** (each names file's, re-read), **G-S (a record, not a stop):** for every unit the row records whether its name string is one of the net's train name strings. 14.4 lists the units that are seen by construction (the R labels under both nets, hyperlink under N_P); **any other unit whose string is a train string is named in the record and kept in the main calls**, and the beside read splits seen from unseen names.
- **G-L (the loader, on WebQSP rows of N_KB):** the stored net, reloaded, reproduces R3's recorded ranks of its stored test edges (300 evenly spaced; name and mean name; ranks agree within max(1, 0.001 x the recorded rank) on at least 99 percent), so the loader and the table are the fit's own. A row that reads N_KB on any target is run after this row passes; the report stops if it is missing or failed.
- **G-0 (N_P, as R3's):** the unfitted N_P scores the cosine on the dev edges (largest score difference at most 1e-4, MRR difference at most 1e-3).
- **G-U:** every unit's test sources are test nodes, every edge's label is the unit's label, and the unit's edge count is recorded.
- **Self-test (toy):** a stored net is saved and reloaded and reproduces the fit's own recorded ranks; the same planted operators read on a second graph make the name net beat the cosine; a different planted operator does not; the report's rule runs on stand-in records.

### 14.8 What this does not test (named, not run)

- **The question level** (an MLP or GNN trained on passage datasets, scored zero-shot on KB datasets, and back). It starts only if links cross (14.5); it needs the verdict, and the links, first.
- **The free-angle ceiling** (R1's own per-corpus operators on the same edges): the share of the target's own gain the transferred operator recovers.
- **Relation signatures from edges instead of names** (a relation's operator from its own edges' statistics, usable where names are few, as on the passage side).
- **A second passage net** (a net fitted on 2Wiki or on several passage corpora at once) and **a KB net fitted without the R labels** (so that no link name of the passage side was seen).
- This is SEM2's, not the journal's work.

### 14.9 Code, rows and outputs

- **outputs/mp_unified/zsem2r6.py** (mine; imports zsem2 and zsem2r3 and edits neither): `selftest`, `names` (R3's names command on a passage corpus), `fit --net p` (N_P), `read --net kb|p --ds D`, `report`. Records carry the code's sha256 and the reads carry the net's file sha256; the report stops on a mismatch.
- **Outputs:** outputs/sem2/r6/nets/p_hotpotqa.npz and .json (the N_P weights and record), outputs/sem2/r6/reads/<net>_<dataset>.json and .npz, then outputs/sem2/r6/report.json and report.md. The json and md are committed, the npz are not.
- **Rows (host, 8 threads; the first reads of N_KB do not wait for N_P):** the names row on HotpotQA and the loader-gate row on WebQSP first; then N_KB on the other six datasets; N_P's fit on HotpotQA (about the length of a short R1 fit); then N_P on the seven. Memory is asked from the measured peaks of the same datasets' R4 rows.

### 14.10 What follows (fixed now)

- **BRIDGES:** the next page is the question level: links produced by a net fitted on one type, built on the other type's graph, entering the pools and the MLP and GNN of that dataset. It is declared before it runs and needs the user's word.
- **ONE_WAY:** the same, in the direction that crossed, and the direction that did not is named.
- **NOT_BRIDGED:** the operators are type-specific as fitted here; the working path stays per-corpus operators (R1) and the stored links, and what is named in 14.8 (signatures from edges, a KB net without R labels, a second passage net) is what could still bridge them.
- Either way: no pool, link or PL change follows from R6, and no MLP or GNN is trained.

## 15. R7 declared: links withheld, does the semantic branch fill what the stored walk cannot reach? (26 November 2026, 00:10 IST, 18:40 UTC), before any R7 number

### 15.1 Why, and what was seen

- **The user, 25 November, on the branch:** "put the walks itself in the semantic space ... a semantic tree or neuron". **R5 (section 13)** read BRANCH_CARRIES on six of six, but against the stored links themselves, where a stored walk scores R-precision 1 by definition. Section 13.8 named the read that says whether a semantic branch is worth anything where links are missing; I offered it and the user said "Declare it" (26 November). **Priority is unchanged: pools and links first.** This read is CPU only, never on the GPU, and its rows go last in the feed, behind every link and pool row.
- **Seen before this page:** R5's records. They hold the per-unit R-precision of each scorer but **not the ranked lists** (the lists were checkpoints, deleted when each row ended), so this read recomputes them with R5's own streaming code. **Not seen:** any number with a link withheld, or any ranked list of R5.
- **Two facts, stated before any number, that bound this read.**
  1. **A walk over withheld links has no false positives.** A path that survives is one of the stored paths, and the exclusions are R5's, so every node it reaches is a true member of T2. The walk's list is shorter than m, and a semantic fill can only add to the slots it leaves empty. The most a fill can add is the scorer's own hit rate times the share of T2 the walk misses. At f = 0.5 that is a few points at most where R5's chain-point reached 2.5 to 4.2 points (MetaQA, WebQSP, HotpotQA, 2Wiki), more on MuSiQue and SQuAD.
  2. **A leak I did not state in section 13.** R1's operators are fitted on edges drawn from the pairs whose two ends are both TRAIN nodes (`zsem2.take(..., is_train, ...)`; the plan's S edges likewise; a draw, so a pair of that kind is one the fit can see, not one it surely saw), and the node split is 10% test, 10% dev, 80% train (`zsem2.node_split`). R5's first hop starts at a test source, so it is clean. **Its second hop does not: the pair (a, b) of a branch a and a second-hop node b is a pair the fit can draw for about 64% of pairs** if the split were independent of the graph (0.8 x 0.8), so chain-point's second step may have been fitted to some of the very pairs it is scored on. chain-cos has no fitted part. This can inflate R5's gap over chain-cos and this page's fill gap alike. **15.5 declares the control that answers it, and a read of R5 under it.**

### 15.2 The question

R5's units (the same sources, labels, T1 and T2, rebuilt from R5's seeds and checked against R5's record). **Withhold a fraction f of the stored edges. The stored walk then reaches part of T2. Fill the rest of the list with a semantic scorer. Does a fill by R1's operators through branches (chain-point) reach more of the rest than a fill by cosine to s?**

### 15.3 Withholding and the stored walk

- **Which edges.** Every stored edge of a chain label (R: the unordered pair, label L; S: the directed edge, label L). An edge is withheld iff u(edge) < f, where u is a fixed 64-bit mix of (stream seed 7, L, the smaller end, the larger end for R or the source, destination for S) read as a number in [0, 1). **f in {0.25, 0.5, 0.75, 0.9}; withheld sets are nested** (a larger f withholds everything a smaller f withholds). f = 0 is the unit's own record (a gate).
- **The stored walk W_f(unit):** the nodes b reached by at least one alive path s to a to b, a in the TRUE T1 (a first-hop edge (s, a) and a second-hop edge (a, b) both alive), minus ({s} union the TRUE T1). **Using the true T1 for the exclusion is the evaluator's exclusion of R5 and gives the walk a small free help** (a T1 node whose own first-hop edge was withheld would otherwise appear as a second-hop answer); it favours the baseline, and so works against the fill.
- **|W_f| <= m always, and W_0 = T2.**

### 15.4 The lists and the measures

- **Semantic lists:** R5's four scorers (cosine, chain-cos, chain-point, compose), the same code, b = 10, K = 400, `--threads 8`, exclusions as R5. They do not depend on f.
- **Combined list C_x(f)** for a scorer x: W_f first (any order: all of it is inside the top m), then x's list with every node of W_f skipped, cut at k. **R-precision** (k = m, main) and **recall at 4m** (k = 4m, beside) as R5. Pure W_f is the list with no fill; pure x is R5's list.
- **Unseen members (the control of 15.1).** T2's node b is **seen** if some a in T1 has an edge (a, b) under L2 with both ends TRAIN nodes (a pair the fit can draw), and **unseen** otherwise. The **unseen measures** are the same R-precision and recall with only unseen members counted in the numerator (the denominator m is unchanged). Every unit and scorer is read both ways.

### 15.5 The calls and the rules (fixed now)

- **Per dataset, main call (f = 0.5, fixed now, not chosen from the results): C_chain-point minus C_cosine, R-precision, mean over the pooled test units, paired bootstrap over units (2,000 resamples, seed 0, `zsem2.paired`).** ABOVE, LEVEL or BELOW. Datasets are the six R5 read; MultiHop-RAG has 0 units under the chain labels and has no row.
- **FILL_CARRIES if the main call is ABOVE on at least 4 of the datasets read and BELOW on none; otherwise NOT_CARRIED.** Fewer than 4 read: INCOMPLETE.
- **Control, declared now and reported with the call:** the same call on the unseen measure. **CONTROL_HOLDS if ABOVE on at least 4 and BELOW on none.** The verdict line says one of FILL_CARRIES (both hold), FILL_CARRIES_SEEN_ONLY (the main call carries, the control does not: the gain is partly the fit's exposure), NOT_CARRIED.
- **R5 under the control:** chain-point minus chain-cos on the unseen measure (it does not depend on f). **R5_CONTROL_HOLDS if ABOVE on at least 4 and BELOW on none.** If it does not hold, a dated note goes under 13.12 saying that R5's call included the exposure of 15.1(2); 13.12's numbers stay as filed.
- **Read beside (decide nothing):** all four f; C_chain-point minus W (does any fill help), minus C_chain-cos, minus C_compose; W alone against chain-point alone by f (the crossover: the smallest grid f at which chain-point alone is at least W alone); recall at 4m; by |T2| (2-3, 4-10, 11-100); the share of T2 the walk reaches (mean |W_f| / m) by f.

### 15.6 Expectations, written before any number (a failed one is reported, not explained away)

- **W alone** falls with f from 100% (f = 0) as the share of surviving paths: at f = 0.5 between about 25% and 70% of T2, under 10% at f = 0.9 on most datasets.
- **The main call:** the fill gain is R5's gap times the share the walk misses, so +0.5 to +1.5 points on the four near-floor datasets and a few points on MuSiQue and SQuAD; **I expect ABOVE on at least five of six, FILL_CARRIES, small.** The intervals shrink with the gain; they may touch 0 on one or two of the four.
- **The control:** the unseen members are about a third of the second-hop edges. **I expect it to hold on SQuAD and MuSiQue and to be weaker on the near-floor four; CONTROL_HOLDS is about even.** This is the expectation I am least sure of.
- **R5_CONTROL_HOLDS:** expected, with smaller gaps than 13.12. If it fails, R5's reading is amended by the note of 15.5.
- **Crossover:** chain-point alone below W alone at every grid f on the four near-floor datasets (their chain-point level is 2.5 to 4.2 points); at f = 0.9 only on SQuAD and MuSiQue.
- **What this will not show whichever way it falls:** a gain of a pool or a retriever. It reads R-precision against stored links with some withheld.

### 15.7 What this does not test (named, not run)

- **Noisy links** (extra or rewired edges): the stored walk then has false positives and a semantic score could re-rank it. That needs a chain-point score for arbitrary nodes, which the top-K lists do not give, so it is a different code path.
- Withholding by hub degree or by relation, the first hop only, the reverse direction, a third hop, other widths, a trained scorer, and any pool or retrieval gain.
- This is SEM2's, not the journal's work.

### 15.8 Gates (a failure stops the row)

- **G-E**, **G-R** as in 12.9 (R1's operators reproduced; `--threads 8`).
- **G-U:** the rebuilt units equal R5's record (`s`, `l1`, `l2`, `grp`, `m`, `n1` of outputs/sem2/r5/<dataset>/e3v.npz, whose sha the record names). Not applied in a smoke run.
- **G-S:** the per-unit R-precision of the four scorers computed from the recomputed lists equals R5's `rp_*` for at least 99% of the units, with a mean absolute difference of at most 0.002. Not applied in a smoke run.
- **G-W:** W_0 equals T2 for every unit; W_f is inside T2 and nested in f for every unit; the hash recomputed by a scalar integer implementation equals the vectorised one on 2,000 edges; W_f recomputed by an independent per-unit scalar walk (R4's `true_sets` and the scalar hash) equals the stored one for 40 units and every f; the withheld share of 200,000 sampled edges is within 1.5 points of f; the seen/unseen flag of 40 units recomputed by a direct scan equals the stored one.
- **Self-test:** a planted toy graph: the hash, the walk, the nesting, the combined list against a brute-force construction, the seen/unseen flag, R-precision by hand, and the report's rule over stand-in records.

### 15.9 Code, rows and outputs

- **outputs/mp_unified/zsem2r7.py** (mine; imports zsem2, zsem2t.operators, zsem2r4 and zsem2r5, read-only; edits none): `selftest`, `run --ds D --host --threads 8 [--smoke]`, `report` (laptop). Records carry the script's sha and R5's; `report` stops on a mismatch.
- **Rows:** `r7smoke-squad` (a code check, read by nothing), then `r7-<dataset>` for the six; 8 CPUs; memory as R5's rows for the same dataset (7, 8, 12, 20, 24, 28 GB); `--threads 8`; **last in the feed, behind the r6 rows**. The feeder retries a failed row once by its own name.
- **Outputs:** outputs/sem2/r7/<dataset>/e3v.json and .npz, then `report` writes outputs/sem2/r7/report.json and report.md.

### 15.10 What follows (fixed now)

- **FILL_CARRIES:** it says a semantic fill reaches a little of what withheld links leave, on R-precision against stored links, and no more; the noisy-links read of 15.7 is declared only on the user's word. **FILL_CARRIES_SEEN_ONLY or NOT_CARRIED:** the fill is closed on R1's operators and the exposure of 15.1(2) is part of the record.
- Either way: no pool, link or PL change follows, and no model is trained.

### 15.11 Records and call (26 November 2026, 03:06 IST, 21:36 UTC)

Read by `zsem2r7.py report` from `outputs/sem2/r7/<dataset>/e3v.json` (every record carries the script's sha, 961539e0..., and R5's, 69d287d0...; the report stops on a mismatch). **Six datasets read; MultiHop-RAG has no unit under the chain labels and no row.** All six rows ran the gates: G-E, G-R (rank agreement 1.0), G-U (rebuilt units equal R5's), G-S (the per-unit R-precision of all four scorers equals R5's on 100% of units, mean absolute difference 0.0) and G-W (hash, walk, nesting, seen flags) PASS on every dataset. R-precision in points, mean over the pooled test units, f = 0.5; intervals are the paired bootstrap over units (2,000 resamples, seed 0).

| dataset | units | walk alone | C_cosine | C_chain-point | C_chain-point minus C_cosine (main call) | control: unseen members | R5 under the control (chain-point minus chain-cos, unseen) |
|---|---:|---:|---:|---:|---|---|---|
| SQuAD | 710 | 35.17 | 42.14 | 46.45 | +4.31 [+3.48, +5.11] ABOVE | +0.58 [+0.26, +0.91] ABOVE | +0.66 [+0.30, +1.08] ABOVE |
| MuSiQue | 900 | 28.54 | 32.24 | 33.59 | +1.35 [+0.77, +1.95] ABOVE | +0.39 [+0.05, +0.75] ABOVE | +0.61 [+0.20, +1.01] ABOVE |
| MetaQA | 1,200 | 26.90 | 28.10 | 28.72 | +0.62 [+0.18, +1.06] ABOVE | -0.05 [-0.23, +0.12] LEVEL | +0.15 [-0.05, +0.34] LEVEL |
| WebQSP | 1,200 | 24.99 | 26.07 | 26.69 | +0.62 [+0.05, +1.21] ABOVE | +0.16 [-0.09, +0.42] LEVEL | +0.30 [-0.00, +0.63] LEVEL |
| HotpotQA | 1,200 | 27.59 | 29.55 | 30.20 | +0.65 [+0.36, +0.93] ABOVE | +0.19 [+0.02, +0.37] ABOVE | +0.35 [+0.15, +0.54] ABOVE |
| 2Wiki | 1,200 | 26.41 | 27.72 | 28.46 | +0.74 [+0.49, +1.01] ABOVE | +0.28 [+0.14, +0.41] ABOVE | +0.54 [+0.38, +0.70] ABOVE |

**Rule 15.5: the main call is ABOVE on 6 of 6 read and BELOW on none. The control (unseen members) is ABOVE on 4 (SQuAD, MuSiQue, HotpotQA, 2Wiki) and BELOW on none: CONTROL_HOLDS. R5 under the control is ABOVE on the same 4 and BELOW on none: R5_CONTROL_HOLDS (so 15.5 asks for no note under 13.12). Verdict: FILL_CARRIES.**

**Size, said plainly.** The ABOVE is small. On the four near-floor datasets the main gain is +0.62 to +0.74 points, and WebQSP's interval starts at +0.05; on the control it is +0.16 to +0.58 points, with HotpotQA's interval starting at +0.02 and MuSiQue's at +0.05, and MetaQA and WebQSP LEVEL. Withholding half the stored chain-label edges leaves the stored walk 25.0 to 35.2 points of R-precision (a full set of links reads 100 by definition); the best fill reaches 26.7 to 46.5. Measured against the walk alone, the fill adds +11.27 (SQuAD), +5.05 (MuSiQue), +1.81 (MetaQA), +1.70 (WebQSP), +2.61 (HotpotQA) and +2.05 (2Wiki) points, which is 17%, 7%, 2.5%, 2.3%, 3.6% and 2.8% of the 64.8, 71.5, 73.1, 75.0, 72.4 and 73.6 points (in the same order) the withholding costs. **A semantic fill recovers a few percent of what withheld links lose on five datasets and about a sixth on SQuAD; it is no substitute for a link a walk needs.**

### 15.12 The reads beside (decide nothing)

- **By f.** The main contrast is ABOVE at every f on SQuAD, MuSiQue, HotpotQA and 2Wiki; MetaQA (+0.20) and WebQSP (+0.12) are LEVEL at f = 0.25 and ABOVE from 0.5. It grows with f: SQuAD +1.52 to +8.85, MuSiQue +0.48 to +2.70, the other four +0.12 to +0.26 at f = 0.25 and +0.95 to +1.50 at f = 0.9. C_chain-point minus the walk alone is ABOVE at every f on all six (+0.88 to +23.83).
- **The walk's reach** falls from 56.3 to 65.0% of T2 at f = 0.25 to 24.99 to 35.17% at f = 0.5, 6.1 to 11.1% at 0.75 and 1.0 to 2.0% at 0.9; the unseen share of T2 is 32.2 to 35.3% on all six.
- **Crossover** (the smallest grid f at which chain-point alone is not BELOW the walk alone): 0.75 on SQuAD and MuSiQue, 0.9 on MetaQA, WebQSP, HotpotQA and 2Wiki (at 0.9 the walk reaches only 1.0 to 2.0% of T2; chain-point alone is then +1.44 to +2.85 above it on the four).
- **Recall at 4m**, C_chain-point minus C_cosine: ABOVE at every f on all six; at f = 0.5 +7.08 (SQuAD), +3.83, +4.09, +3.17, +2.08, +2.47. On the unseen measure recall at 4m is ABOVE at every f on all six (+0.60 to +1.48 at f = 0.5), including MetaQA and WebQSP, where the R-precision control is LEVEL.
- **Other fills at f = 0.5:** chain-point minus chain-cos is ABOVE on 6 of 6 (+0.66 to +3.58); minus compose is ABOVE on SQuAD, MuSiQue, HotpotQA and 2Wiki, LEVEL on MetaQA (+0.34) and WebQSP (-0.41 [-0.87, +0.01]; the point estimate is below zero).
- **By |T2| at f = 0.5:** the gain sits in the largest sets (11 to 100 members) on five of six (SQuAD's 4 to 10 bin is ABOVE too); MetaQA's ABOVE is in its smallest bin (2 to 3, +1.16) and its 11 to 100 bin is LEVEL (+0.36 [-0.02, +0.73]). The sets of 2 to 3 members are LEVEL on the other five.

### 15.13 Expectations against the outcome (15.6; a failed one is reported, not explained away)

- **W alone** at f = 0.5 was expected between about 25% and 70% of T2 and under 10% at f = 0.9: **held**, but at the bottom of my range (25.0 to 35.2%) and well under 10% at 0.9 (1.0 to 2.0%); my 70% was loose (two surviving edges at f = 0.5 is 25% per path).
- **The main call:** expected +0.5 to +1.5 on the four near-floor datasets, a few points on MuSiQue and SQuAD, ABOVE on at least five of six: **held** on the four (+0.62 to +0.74) and on SQuAD (+4.31); **MuSiQue's +1.35 is below "a few points"**; ABOVE on six of six. No interval touched 0, the nearest being WebQSP's +0.05.
- **The control:** expected to hold on SQuAD and MuSiQue and to be weaker on the near-floor four, CONTROL_HOLDS about even: **held** (4 ABOVE), and **two of the four (HotpotQA, 2Wiki) were ABOVE where I expected weakness**; MetaQA and WebQSP were LEVEL. The sizes (+0.16 to +0.58) are at the floor of the scale.
- **R5_CONTROL_HOLDS:** expected, with smaller gaps than 13.12: **held**; the gaps are +0.15 to +0.66 against 13.12's +1.12 to +7.68, that is 9% (SQuAD), 19% (MuSiQue), 8% (MetaQA), 27% (WebQSP), 26% (HotpotQA) and 32% (2Wiki) of the filed gap. The unseen members are about a third of T2, so a gain spread evenly over the members would leave about a third: **on SQuAD, MuSiQue and MetaQA most of R5's gap sits on second-hop pairs the fit could draw** (15.1(2)); on the other three it is roughly proportional.
- **Crossover: failed.** I expected chain-point alone below the walk alone at every grid f on the four near-floor datasets, and to cross at f = 0.9 only on SQuAD and MuSiQue. The four cross at f = 0.9 (where the walk reaches 1 to 2% of T2), and SQuAD and MuSiQue cross earlier, at 0.75.

### 15.14 What this says, and what it does not

- **Says:** on R-precision against stored links with half withheld, a fill by R1's operators through branches reaches a little more of the missing second hop than a cosine fill, on all six datasets (+0.6 to +4.3 points) and, on the control that drops the pairs the fit could draw, on four of six (+0.2 to +0.6). It recovers 2 to 17% of what the withholding costs.
- **Does not say:** that a pool, a retriever or a trained model gains; that links can be dropped because a fill replaces them (it does not: 15.11); that the gain is independent of the fit's training edges beyond the control. R1's operators were fitted on the full graph, **including the edges this read withholds**, and R7 never refits without them; the unseen control removes the members for which any pair the fit could draw reaches them, which is the cleaner read, and its gains are the +0.16 to +0.58 above. A refit on the surviving edges only is the clean test, named, not run.
- **Bearing on the link work:** link tightening that drops a link carrying a gold path loses it from the walk, and a semantic fill gets back a few percent of that; the layout stays as the link line decides it.

### 15.15 What this closes, and disclosed

- **Closes:** the withheld-links read of 13.8, against the expectation scorecard above. FILL_CARRIES is a small, floor-size result. **Nothing is queued by this page.** The noisy-links read (15.7) and a refit on the surviving edges are declared only on the user's word. No pool, link or PL change follows and no model is trained (15.10).
- **Disclosed:** f = 0.5 and the call were fixed before the number (15.5); the other f are beside. The walk uses the true T1 for its exclusion, which favours the baseline (15.3). One expectation (crossover) failed and two were looser than the outcome (W alone's range, MuSiQue's size). WebQSP's main-call interval and the HotpotQA and MuSiQue control intervals start within 0.05 points of zero.

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
