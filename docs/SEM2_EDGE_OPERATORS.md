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

### 14.11 Records and call (26 November 2026, 03:53 IST; 22:23 UTC)

Read by `zsem2r6.py report` from `outputs/sem2/r6/reads/<net>_<dataset>.json` and `nets/p_hotpotqa.json` (every record carries the script's sha, 8b58d91f..., and the nets' file shas; the report stops on a mismatch; **missing rows: none**). Gates: G-E and G-N PASS on every row; G-L PASS (the stored N_KB, reloaded, reproduces R3's recorded ranks on 300 stored test edges: name and mean-name ranks within the tolerance on 100% and equal on 98.7% and 96.0%); G-0 PASS (N_P unfitted scores the cosine: largest score difference 4.8e-7, MRR difference 1.5e-9); G-U PASS; G-S recorded (the seen names: N_KB's three R labels on the passage units, N_P's four on HotpotQA; MetaQA's two R labels under N_KB, WebQSP's three R labels under N_P: beside, not in the main calls). N_P's fit: dev MRR 7.80 at the start (the cosine), best 12.67 at epoch 8 of 10 (ended by patience); the four train names are R.mention, R.name_comention, R.title_comention, S.hyperlink. MRR in points on test sources; intervals are the bootstrap over units (2,000 resamples, seed 0); one WebQSP unit (S.form of government countries) had fewer than 20 test edges and is dropped from direction 3 (162 of the 163 held-out relations read, plus MetaQA's 7).

| direction | units | Delta name - cosine | Gain from the name | call | mean MRR cosine / mean name / name / shuffled |
|---|---|---|---|---|---|
| 1 KB -> KB (WebQSP net on MetaQA's S relations) | 7 | +0.79 [-0.21, +2.47] LEVEL | -0.13 [-1.51, +1.35] LEVEL | NO | 0.43 / 1.36 / 1.22 / 0.48 |
| 2 KB -> passage (WebQSP net on the 15 passage units) | 15 | -1.37 [-1.89, -0.85] BELOW | -0.80 [-1.43, -0.36] BELOW | LOSES | 8.90 / 8.34 / 7.54 / 7.88 |
| 3 passage -> KB (HotpotQA net on WebQSP's 163 and MetaQA's 7) | 169 | +1.53 [+0.68, +2.52] ABOVE | +0.15 [-0.08, +0.38] LEVEL | GENERIC | 2.52 / 3.90 / 4.05 / 4.03 |
| 4 passage -> passage, control (HotpotQA net on the other four corpora) | 11 | +2.36 [+1.06, +3.87] ABOVE | +0.15 [-0.39, +0.70] LEVEL | GENERIC | 9.28 / 11.49 / 11.64 / 10.45 |

**Rule 14.5: the direction 4 control is ABOVE (+2.36), so the cross-type reads are about types. Direction 3 (passage to KB) has Delta ABOVE and direction 2 (KB to passage) has Delta BELOW: verdict ONE_WAY (passage -> KB).** The call of direction 3 is **GENERIC**: the name adds +0.15 [-0.08, +0.38] over the net's relation-blind operator (LEVEL), and a name taken from another relation does as well as the right one (name minus shuffled name +0.02 [-0.10, +0.15]).

**Size, said plainly.** The gain is at the floor of the scale. On direction 3 the cosine reads 2.52 MRR points and the generic operator 3.90; WebQSP's 162 relations dominate the 169 units. On MetaQA alone (7 relations) the cosine reads 0.43 and the net 1.89 (Delta +1.45 [+0.16, +3.73], GENERIC). Where the net's own labels are read on new nodes (HotpotQA, in sample, 4 units) the cosine reads 7.86 and the net 11.98 (Delta +4.12, the name adding +1.04 [+0.32, +1.87]: TRANSFERS, the only row where the name matters). **What crosses from passages to a KB is one generic link operator worth about 1.5 MRR points on relations the net never saw, not a relation-specific map.**

### 14.12 The reads beside (decide nothing)

- **Per direction, mean-name and shuffled-name minus cosine.** Direction 1: +0.92 [+0.12, +1.93] ABOVE and +0.05 [-0.46, +0.48] LEVEL. Direction 2: -0.56 [-1.12, -0.06] BELOW and -1.02 [-1.66, -0.45] BELOW. Direction 3: +1.38 [+0.52, +2.33] ABOVE and +1.51 [+0.69, +2.46] ABOVE. Direction 4: +2.21 [+0.87, +3.67] ABOVE and +1.17 [+0.39, +1.99] ABOVE (name minus shuffled +1.19 [+0.16, +2.38] ABOVE: among passage corpora the name does carry a little).
- **Units with a gain.** Relations with d above 0: direction 1 5 of 7; direction 2 1 of 15; direction 3 109 of 169; direction 4 11 of 11.
- **Direction 1 (KB to KB).** The WebQSP net on MetaQA's S relations: Delta +0.79 [-0.21, +2.47] LEVEL with the cosine at 0.43 (the floor R3 found); the generic operator is ABOVE the cosine (+0.92 [+0.12, +1.93]) but the name is not above the generic one.
- **Direction 2 (KB to passage).** BELOW on the pooled 15 units and on the 13 units whose names the net has seen (R labels: Delta -1.53 [-2.07, -0.96] LOSES, one of 13 units above zero): **the loss is not caused by unseen names.** Per corpus: SQuAD, MuSiQue and HotpotQA BELOW, 2Wiki LEVEL, MultiHop-RAG has one unit (TOO_FEW_UNITS).
- **S.hyperlink under N_KB** (a passage-side name the net has never seen; two units, 200 edges each, MRR points): HotpotQA cosine 7.41, name 7.14, mean name 7.92, shuffled 7.23; 2Wiki 6.67, 6.32, 6.95, 6.96. The name is below the cosine on both and the generic operator is above it on both (+0.51, +0.28), a read of two units.
- **By dataset under N_P** (beside, the units of each corpus): ABOVE on SQuAD (+3.09), MuSiQue (+0.58), MetaQA (+1.45), WebQSP (+1.53), HotpotQA (+4.12, in sample), 2Wiki (+3.64); GENERIC on all but HotpotQA; WebQSP's three R labels LEVEL (-0.37 [-3.64, +2.53], three units).

### 14.13 Expectations against the outcome (14.6; a failed one is reported, not explained away)

- **X22 (KB to KB, Delta not ABOVE): held** (+0.79, LEVEL; MetaQA at the floor).
- **X23 (KB to passage, Delta not ABOVE, and the hyperlink units no better than cosine): held** (BELOW at -1.37; the name is below the cosine on both hyperlink units).
- **X24 (passage to KB, Delta not ABOVE): FAILED.** Delta is +1.53 [+0.68, +2.52] ABOVE. I had written that if anything were above it would be the mean name, reading GENERIC: **that half held** (the call is GENERIC and the name adds nothing), but I had expected it not to reach ABOVE, and it did, on 162 WebQSP relations and 7 MetaQA ones.
- **X25 (passage to passage, Delta ABOVE): held** (+2.36). **X26 (N_P in sample, Delta ABOVE): held** (+4.12).
- **My expectation for the verdict (NOT_BRIDGED): FAILED.** The verdict is ONE_WAY (passage to KB), by the Delta rule of 14.5; the call says what crosses is generic. The working idea (the user's, 25 Nov) was that link operators bridge passage corpora and KBs: **one generic operator does, in one direction; a relation-specific one does not, in either.**

### 14.14 What this says, and what it does not

- **Says:** a net fitted on one passage corpus's four link labels produces an operator that places the true tails of KB relations above plain cosine (+1.53 MRR points on 169 relations, 109 of them above zero), and the relation's name does not matter for that (a shuffled name does the same); the reverse does not hold (a net fitted on a KB's relations loses to cosine on passage links, -1.37, and on the names it has seen too). Passage corpora carry each other's operators (+2.36, with a small name effect beside).
- **Does not say:** that a pool, a retriever, an MLP or a GNN gains; that the names mean anything across types (they do not here); that this is the only route (relation signatures from edges and a KB net fitted without R labels are named in 14.8, not run); that the generic operator is not simply "links go to a near neighbour in a certain direction" learned from any corpus (no read separates the two); that the size matters: the cosine's own MRR on the KB units is 0.4 to 2.5 points.
- **Bearing on the user's idea (25 Nov: passage to KB and back, 'if we can transfer links only then we can work with them during questions with mlp and gnn').** At link level, the passage-to-KB direction crosses as one shared operator, the KB-to-passage direction does not, and no relation-specific map crosses in either. The question-level read of 14.10 (links produced by a net fitted on passages, built on a KB graph, entering the pools and the models) **is declared only on the user's word**; no pool, link or PL change follows from R6.

### 14.15 What this closes, and disclosed

- **Closes:** R6's four directions against 14.5's rule. **Nothing is queued by this page.** 14.8's reads (the free-angle ceiling, signatures from edges, a second passage net, a KB net without R labels) are declared only on the user's word.
- **Disclosed:** the calls and the verdict rule were fixed before any number (14.5). X24 and my verdict expectation failed. Direction 3's units are dominated by one KB (162 of 169 WebQSP); MetaQA alone is at the floor; the hyperlink units are two each. HotpotQA is the only passage net fitted. The generic and the shuffled operators are the net's own, so GENERIC is a statement about this net: a relation-blind operator is enough to explain the gain here, which is not a proof that no relation-specific signal exists in the data.


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

## 16. R8 declared: does routing the stored walk by the question move the node bound? (26 November 2026, 04:13 IST, 22:43 UTC), before any R8 number

### 16.1 Naming, why, and what was seen

- **Naming.** The user's 26 November message was "work on r7 as well based on the 5 points you told me". R7 (section 15, withheld links) is already filed (15.11), so I read it as **the round after R7, designed from the five points of the last answer**: what limits retrieval; what transfers is one generic operator; what may be missing is question-conditioned routing, walk-level self-supervision, composition across hops and a trainable adapter; the rules are too rigorous; the objective may be wrong. This page is called R8. The same message set the form: **constrained programming, not a linear program**, wherever a selection or a budget appears. 16.4 states the programs, their variables, objective and constraints, and which solver is exact.
- **The point read here: question-conditioned routing.** Why it, and not another:
  - R2a (9.12): choosing the best operator per seed has no room at fused R@5 (C = oracle minus trav, -0.02 to +0.04 points). R5 (13.12) and R7 (15.11) read BRANCH_CARRIES and FILL_CARRIES, both at the floor of the scale. All three score **rank quality at the top of a list** (R@5, R-precision). None scores the quantity the pool bound is stated in: **which golds enter a pool of a stated number of nodes**.
  - UL4 (docs/UL4_UNIVERSAL_LP.md section 10): the factor lies in per-question depth (oracle Q-min at ALL 0.95: MuSiQue 3,382, MetaQA 2,804, WebQSP 10,250 mean nodes against the 2,500 the user named). No router has been read at that quantity.
  - Other sessions read the cross-question allocation (a depth signal) and the patterns of gold links. **Neither is read here.** R8 reads the within-question half: for one question and its seeds, which stored links earn their node.
- **Seen before this page:** R1 to R7's records, and UL4's depth bound (counts only). **Not seen:** any gold coverage of any ordering of a stored walk's reach. No R8 number exists.

### 16.2 The question, in two parts

- **Part A (no scorer, no parameter): the room.** Of the questions whose first stage misses a gold, how many could a two-hop stored walk from the first stage's seeds complete at all, and at how many new nodes? That is the room any router has, perfect or not.
- **Part B (only where Part A finds room): the router.** At 1,500 new nodes, how many of those completable questions does a question-conditioned router complete, against the same program with a question-only score, an operator-only score, a shuffled question and a random order?

### 16.3 Questions, first stage, walk, pool (fixed now)

- **Questions.** The selection carve: s1sel (zul1.carve_questions) on SQuAD, MuSiQue, MetaQA, HotpotQA and 2Wiki; MultiHop-RAG's dev split (224); **WebQSP: PL's dev (outputs/pl2/webqsp/pop_dev.npz, as UL3's code reads it; its ids are disjoint from s1eval's, gate G-Q)**. Every carve question with at least one gold. **No s1eval question is read.** e3v only; the frozen arm is not run.
- **First stage.** SEM1's, unchanged: v = zenc's exact dense top 1,000 on e3v; SPLADE's top 1,000; base = RRF(v, SPLADE) with C = 60, **depth F = 1,000** (zsem1.rrf). F-list(q) is the base list's nodes. A corpus smaller than F (MultiHop-RAG, 609 nodes) gives a list that ends early and holds the corpus. **cov0 = the share of questions with every gold in F-list(q)** (the first stage's ALL-gold coverage).
- **Seeds.** The first **b = 25** nodes of the base list.
- **The walk.** Two hops over the stored links R1 gave an operator (every label with an operator: R labels in both directions, S edges in their stored direction, rare S relations through the pooled operator, as 9.1). N+(x) is x's stored out-neighbours. **Hop 1: A(q) = N+(seeds) minus the seeds. Hop 2: D(q) = N+(the hop-1 nodes that are in the pool) minus the nodes already in the pool or in A(q).** The K store (the encoder's own neighbours) is not followed; no third hop (16.7).
- **Pool and cost.** pool = F-list(q) plus the nodes a program adds. **Cost = the number of added nodes**: a node already in the F-list is free, as a pool member and as a parent. **M' = the added-node budget; the total is F + M' = 2,500 at M' = 1,500** (the user's bound). The ladder is M' in {25, 100, 500, 1,500}; **the main read is M' = 1,500**.

### 16.4 The constrained programs (variables, objective, constraints, solver)

For a question q let C(q) = (A(q) union D(q)) minus F-list(q) be the candidates. **Variables:** x_u in {0, 1} for u in C(q): the node u is added to the pool. Let p_a = 1 if a is in F-list(q) and p_a = x_a otherwise (a is in the pool).

- **Closure (the walk's precedence).** For u in A(q): none (a seed is its parent and is always in the pool). For u in D(q) not in A(q): x_u <= the sum over its parents a in A(q) (a stored edge a to u) of p_a. If a parent is in the F-list, u is free to take.
- **Budget.** The sum of x_u <= M'.
- **Pi_oracle(q): minimise** the sum of x_u **subject to** x_g = 1 for every missed gold g (a gold not in the F-list) that is in C(q), and the closure. It is **feasible iff every missed gold is in C(q)**; its optimum is cost*(q). **q is completable at M' iff Pi_oracle is feasible and cost*(q) <= M'.** A question with no missed gold has cost 0 and is not counted as completable (the first stage already covers it).
- **Pi_score(q, w): maximise** the sum of w_u x_u **subject to** the closure and the budget, with a score-derived weight w_u >= 0 (16.6). **The number of branches the walk opens, the sum of x_a over the hop-1 nodes, is not a parameter: it is an output of the program, per question.**
- **Solver and its exactness.** Each is a 0-1 integer program with a precedence (closure) structure; with a cardinality budget the closure-knapsack is NP-hard in general, and these instances are small (Pi_oracle: the missed golds and their parents; Pi_score: at most 5,100 variables, 16.6). **Exact solver: HiGHS branch and bound (scipy.optimize.milp, relative MIP gap 0, time limit 60 s per instance).** An exit that is not proven optimal is counted, listed and gated (G-P); it is never read as optimal. The LP relaxation of every instance is solved as well and the share of instances with LP = IP is recorded (how far the integer program is from a linear one). The cross-question allocation (a total budget over the questions, the multiple-choice knapsack on the per-question curves) is **not** solved here; R8 writes each question's cost*(q) to its npz as an input for it.

### 16.5 Part A: the room (no scorer, no parameter)

- **Measures per dataset:** the number of questions; cov0; for each missed gold, whether it is in A(q), in D(q) only (reached through a hop-1 node), or **beyond** (neither within two hops); the completable share at each M' of the ladder; **H(M') = (the completable share at M') x (1 - cov0)**, the first stage's ALL-gold coverage gain in points if a perfect router spent M'; the median and mean of cost*(q) over completable questions; the field: |A(q)| and the sum over A(q) of the out-degree of each node (an upper bound on the hop-2 frontier, so on what an unordered two-hop walk would add).
- **ROOM rule (fixed now, R2a's line): a dataset has ROOM iff H(1,500) >= 1.0 point.** Reported with a bootstrap interval over questions (2,000 resamples, seed 0) beside; **ROOM is decided on the point.**

### 16.6 Part B: the router (declared now; run only on the ROOM datasets; its code is written and pinned after Part A is filed, and its rule is this page's)

- **Population.** The questions completable at M' = 1,500 (16.4), a seeded draw (numpy default_rng(8)) of at most 600 per dataset.
- **Candidates per scorer.** Scorer x gives every node of A(q) a value; **A_cap = the 100 best hop-1 nodes of A(q) minus the F-list by x** (hop-1 nodes already in the F-list are in the program as free parents); for each hop-1 node in A_cap (or the F-list) its **50 best stored out-neighbours by x** make D_cap. **Caps differ from Pi_oracle's (none); the share of populations whose missed golds all lie in the capped candidate set is reported as the candidate recall of x.** The weight is w_u = value(u) minus the smallest value among the question's candidates, so w_u >= 0 and ties are the scorer's own.
- **Scorers (parameter-free; x = P v / |P v|; R1's operators as stored, G-R):**

| scorer | value of a candidate u reached from its parent a (a seed for hop 1) |
|---|---|
| W0 qcos | x_q . x_u (the question's own geometry, parent-blind) |
| W1 lcos | x_a . x_u (cosine along the stored link; no question, no operator) |
| W2 plaus | the operator of the stored edge's label applied to x_a, dotted with x_u (R1's score of that edge; no question). With several edges a to u, the maximum |
| W3 routed | plaus + lambda x_q . x_u (R2a's form, the stored label as the route; **lambda = R2a's chosen lambda for trav on that dataset**, read from outputs/sem2/r2/read/e3v/<dataset>.json, never refit) |
| W4 routed, another question | W3 with x_q replaced by the question of a fixed derangement of the population (seeded) |
| W5 random | i.i.d. U(0, 1), seeded (an unordered walk's expectation inside the same caps) |

- **Measures at each M' of the ladder:** completion = 1 iff every missed gold of the question is in Pi_score's solution (parents included); the completion rate over the population. Beside: the recovered share of missed golds; the number of branches opened; the cost used; the candidate recall.

### 16.7 What this does not test (named, not run)

- A **third hop** (composition across hops, the point of the five that needs it), more or fewer than 25 seeds, other caps, the K store.
- **A trained router** (a fit on questions, a walk-level self-supervision, an adapter): R8 fits nothing.
- **Any pool, retriever or model gain.** R8's pool is the F-list plus the program's nodes, **not PL's cells** (their walks start from up to 400 seeds and spend the nodes by their own rules); R8's counts are not those of PL's cells and no number of R8 replaces PL's.
- The **cross-question allocation** (16.4), s1eval, and any change of link or layout. This is SEM2's work, not the journal's.

### 16.8 Calls and the verdict (fixed now)

- **Part A:** per dataset ROOM or NO_ROOM (16.5).
- **Part B, main call per ROOM dataset (M' = 1,500): W3 - W0, completion rate in points, over the population, paired bootstrap over questions (2,000 resamples, seed 0, zsem2.paired).** ABOVE, LEVEL, BELOW.
- **Control, declared with the call: W3 - W4** (the question's own contribution): CONTROL_HOLDS on a dataset iff ABOVE there.
- **Verdict:**
  - **ROUTING_CARRIES** iff at least two datasets have ROOM, the main call is ABOVE on at least two thirds of the ROOM datasets and BELOW on none, and the control holds on every dataset where the main call is ABOVE.
  - **ROUTING_NO_QUESTION** iff the main call carries and the control does not (the gain is not the question's).
  - **NOT_CARRIED** otherwise. **INCONCLUSIVE** if fewer than two datasets have ROOM (only A is then filed).
- **Beside (decide nothing):** W3 - W2, W3 - W1, W3 - W5, W0 - W5, every contrast at M' = 25, 100, 500; the implied ALL-gold coverage gain over the carve (completion rate x H); the share of the oracle's room closed (the completion rate itself); by the number of missed golds (1, 2, 3 or more).

### 16.9 Expectations, written before any number (a failed one is reported, not explained away)

- **XA1.** ROOM on at least four of the seven. **No ROOM on MultiHop-RAG** (the corpus has 609 nodes, F = 1,000, so cov0 = 1 by construction) **and none on SQuAD** (an answer sits in one paragraph; the first stage at depth 1,000 should hold nearly every gold).
- **XA2.** On the KB datasets (MetaQA, WebQSP) **most missed golds lie beyond two hops of the top-25 seeds** (the first stage barely finds anything there: R2a's base R@5 is 0.45 and 4.56 percent), so the completable share is small and the room is reach, not order.
- **XA3.** cost*(q) is small (**median at most 10 new nodes**): a perfect router needs a handful of nodes; and the field is large: **the median unordered cost (hop-1 nodes plus the hop-2 frontier bound) exceeds 1,500 on every dataset with ROOM.** So the difficulty is choosing, not budget.
- **XB1.** Random (W5) completes under 5 percent; **qcos (W0) completes 20 to 60 percent of the population; W3 - W0 is LEVEL or a small ABOVE (at most +5 points). My expectation is NOT_CARRIED or ROUTING_NO_QUESTION:** R2a's attention added hundredths of a point, and the stored label already routes. **The control (W3 - W4) will hold**, because any real question beats another question's vector, so it only says the question is read at all.
- **XB2.** The branches the program opens per question (Sum x_a) vary widely: a median under 20 and a long tail (a few questions need over 100).

### 16.10 Gates (a failure stops the row), code, rows and outputs

- **G-E:** the graph's family counts equal UL1's grade; the labels followed are R1's operator labels (names equal R1's record). **G-Q:** the carve's ids in order, golds in range, **base5 equals R2a's selection record's base5 within 0.0005** (not WebQSP, which has none: its ids are checked against s1eval's instead).
- **G-A:** for 40 questions' seeds the forward adjacency, and for 40 questions' missed golds the backward adjacency, equal a direct scan of the graph's keys.
- **G-P:** each Pi_oracle solution satisfies the closure and the budget, and cost*(q) lies between the number of reachable missed golds and that number plus the golds reached only through a hop-1 node; **at least 99.5 percent of the instances are proven optimal** (the rest are listed and dropped from the calls, never read as optimal); each LP optimum is at most the IP optimum. **G-R (Part B):** R1's operators reproduced, `--threads 8`.
- **Self-test:** a toy graph with planted two-hop golds: HiGHS equals a brute-force enumeration of the closure-knapsack on 200 random small instances (both programs); the oracle, the walk, the cost and the closure by hand; the report's rule on stand-in records.
- **Code:** outputs/mp_unified/zsem2r8.py (mine; imports zsem2, zsem2t, zsem1, zenc and zul1 read-only; edits none): `selftest`, `reach --ds D --host --threads 8 [--smoke]` (Part A), then `route` (Part B, after Part A is filed) and `report`. Records carry the script's sha.
- **Rows:** `r8smoke-squad` (a code check, read by nothing), then `r8a-<dataset>` for the seven, CPU only, placed in the feed before the r6 rows (the GPU is not ours); memory sized from R5's measured peaks for the same dataset; **priority is unchanged: pools and links first.**
- **Outputs:** outputs/sem2/r8/<dataset>/reach.json and .npz (per question: cost*, completable flags, missed-gold classes, the field), then report.json and report.md; the json and md are committed, the npz is not.

### 16.11 What follows (fixed now)

- **No ROOM on any dataset:** R8 stops at Part A. It says the stored walk's reach, not its order, bounds the pool: no router over two hops of these links can move the bound. What is left is named, not queued: a third hop, and links the rules do not give (R6's and R7's operators as a link source), on the user's word.
- **ROOM on some:** Part B on those, as 16.6 and 16.8.
- **ROUTING_CARRIES:** a full-carve (s1eval) read on those datasets would be declared on the user's word. **Either way no pool, link or PL change follows, no model is trained, and the cross-question allocation stays with its own line.**

### 16.12 Part A filed: ROOM on five of the seven datasets (26 November 2026, 08:15 IST, 02:45 UTC)

- **Records.** `outputs/sem2/r8/<dataset>/reach.json` and `.npz` for the seven, `report.json`, `report.md`. Code `outputs/mp_unified/zsem2r8.py` (committed da5ae385, sha256 d02b42917add...); **all seven records carry that one sha**. Six datasets ran as host rows (`r8a-<dataset>`, 8 cpus, `--threads 8`, 26 November 04:25 to 04:53 IST). **MultiHop-RAG ran on the laptop** (609 nodes, 0.9 s, same code and gates): 16.10 listed seven host rows, so this is a departure from the text, and it changes nothing a number depends on. The npz files are not committed (16.10).
- **Gates, all seven: G-E, G-Q, G-A, G-P PASS.** G-P: every instance proven optimal (`not_optimal` is 0 on all seven); the instances that needed the branch and bound (the others are closed by the closure alone) were 480 on MetaQA, 75 on MuSiQue, 79 on WebQSP, 7 on HotpotQA, 13 on 2Wiki, 3 on SQuAD; LP = IP on 98.8 percent of them on MetaQA, 98.7 percent on WebQSP and 100 percent elsewhere; every solution satisfied the closure and the budget and every cost* lay inside its bounds.
- **The read** (H in points of ALL-gold coverage of the carve; cost* over the completable questions; |A| and the unordered cost are medians over the carve's questions):

| dataset | questions | cov0 | missed golds: hop 1 / hop 2 / beyond (%) | completable of the questions with a missed gold | H(25) | H(100) | H(500) | H(1,500) [95%] | cov0 + H(1,500) | call | cost* median / mean / p90 | hop-1 field \|A\| | unordered cost |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---|---|---:|---:|
| MultiHop-RAG | 224 | 100.00 | 0.0 / 0.0 / 0.0 | 0 of 0 | 0.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 100.00 | NO_ROOM | - / - / - | - | - |
| SQuAD | 1,498 | 99.60 | 33.3 / 50.0 / 16.7 | 5 of 6 | 0.33 | 0.33 | 0.33 | 0.33 [0.07, 0.67] | 99.93 | NO_ROOM | 2.00 / 1.60 / 2.00 | 1,695 | 275,857 |
| MuSiQue | 1,534 | 65.38 | 64.1 / 31.1 / 4.8 | 503 of 531 | 32.79 | 32.79 | 32.79 | 32.79 [30.64, 35.20] | 98.17 | ROOM | 1.00 / 1.28 / 2.00 | 1,005 | 527,825 |
| MetaQA | 1,497 | 18.70 | 14.6 / 53.9 / 31.5 | 994 of 1,217 | 63.73 | 66.33 | 66.40 | 66.40 [63.99, 68.81] | 85.10 | ROOM | 2.00 / 5.63 / 13.00 | 229 | 46,924 |
| WebQSP | 1,507 | 25.15 | 83.5 / 15.6 / 0.9 | 1,108 of 1,128 | 70.07 | 73.46 | 73.52 | 73.52 [71.33, 75.71] | 98.67 | ROOM | 2.00 / 5.21 / 11.00 | 14,862 | 166,404,092 |
| HotpotQA | 1,508 | 83.82 | 95.1 / 4.0 / 0.8 | 242 of 244 | 16.05 | 16.05 | 16.05 | 16.05 [14.19, 17.90] | 99.87 | ROOM | 1.00 / 1.04 / 1.00 | 514 | 3,423,192 |
| 2Wiki | 1,496 | 47.66 | 98.3 / 1.6 / 0.1 | 782 of 783 | 52.27 | 52.27 | 52.27 | 52.27 [49.67, 54.68] | 99.93 | ROOM | 1.00 / 1.29 / 2.00 | 942 | 8,863,648 |

- **Calls, with the size of each.** **ROOM** on MuSiQue (H = 32.79 points, interval 30.64 to 35.20), MetaQA (66.40, 63.99 to 68.81), WebQSP (73.52, 71.33 to 75.71), HotpotQA (16.05, 14.19 to 17.90) and 2Wiki (52.27, 49.67 to 54.68): each far above the 1.0-point line, and each interval clear of it. **NO_ROOM** on SQuAD (H = 0.33, 0.07 to 0.67: the whole interval is below the line, cov0 is 99.60) and on MultiHop-RAG (H = 0.00: cov0 is 100.00 because the corpus of 609 nodes is smaller than F = 1,000).
- **Expectations, scored against the numbers.**
  - **XA1 held.** ROOM on five of seven (at least four expected); none on MultiHop-RAG and none on SQuAD.
  - **XA2 FAILED.** I expected most missed golds on MetaQA and WebQSP to lie beyond two hops. They do not: on MetaQA 14.6 percent are one hop, 53.9 percent two hops and 31.5 percent beyond; on WebQSP 83.5 percent are one hop, 15.6 percent two hops and **0.9 percent beyond**. The completable share is large, not small: 994 of the 1,217 MetaQA questions with a missed gold (81.7 percent) and 1,108 of the 1,128 WebQSP questions (98.2 percent). "The room is reach, not order" is wrong on both KB datasets: the golds sit next to the seeds. I had read R2a's base R@5 (0.45 and 4.56 percent) as the first stage finding nothing there; the first stage's coverage at depth 1,000 is 18.7 and 25.2 percent and the golds it misses are mostly neighbours of its top 25.
  - **XA3 held.** cost* median is 1 to 2 added nodes on every ROOM dataset (at most 10 expected; p90 is 1 to 2 on MuSiQue, HotpotQA and 2Wiki, 11 on WebQSP and 13 on MetaQA), and the median unordered cost is between 46,924 (MetaQA) and about 166 million (WebQSP), against the 1,500 bound, on every ROOM dataset.
- **What the numbers say and what they do not.** The room is the first stage's ALL-gold coverage if a router already knew the gold: from 65.38 to 98.17 on MuSiQue, 18.70 to 85.10 on MetaQA, 25.15 to 98.67 on WebQSP, 83.82 to 99.87 on HotpotQA and 47.66 to 99.93 on 2Wiki, at most 1,500 added nodes. **H is not a gain.** It says where a router could matter; it does not say any router reaches it. Three things stand out: **(1) the budget never binds for a perfect router:** H(25) equals H(1,500) on MuSiQue, HotpotQA and 2Wiki, and MetaQA and WebQSP gain 2.6 and 3.4 points from 25 to 100 added nodes and under 0.1 after 100; **(2) the oracle's walk is one edge to nearly every gold:** cost* per missed gold is 1.01 to 1.17 on the ROOM datasets (a hop-1 parent has to be paid for on 48.3 percent of the completable MetaQA questions, 14.9 percent on MuSiQue, 7.1 on WebQSP, 2.9 on HotpotQA and 1.7 on 2Wiki); **(3) the difficulty is the field, not the budget:** one gold has to be chosen among 229 (MetaQA) to 14,862 (WebQSP) hop-1 nodes, and the unordered two-hop walk would add 10^4 to 10^8 nodes. Part B is that choice.
- **Beside, not a call.** The hop-1 field is large because the R links (mention, name and title co-mention) and the S links have high degree; a gold reached in one hop by an oracle is a neighbour among hundreds to thousands. The unordered-cost column is the counterweight to the H column.

### 16.13 Amendments before any Part B number (26 November 2026, 08:15 IST, 02:45 UTC)

Written after Part A is filed and after the code of Part B is self-tested, **before any Part B number exists** (no Part B run, smoke or full, has been made). The declared rules of 16.6 and 16.8 are not changed; 16.13 states what 16.6 and 16.8 did not and adds beside readings.

- **(a) Where the code lives.** 16.10 put `route` in `zsem2r8.py`. Its sha is in the seven Part A records, so it is not edited. **Part B is `outputs/mp_unified/zsem2r8b.py`** (`selftest`, `route --ds D --host --threads 8 [--workers 6] [--smoke]`, `report`) **and the exact solver with its worker loop is `outputs/mp_unified/zsem2r8w.py`**; both import the earlier modules read-only and edit none. Records carry both shas. **Why a worker module:** HiGHS holds the Python lock (four threads gave 1.2 times the speed of one), so the solves run in worker processes; the programs, the solver, the gap (0) and the time limit (60 s) are as declared.
- **(b) The programs, as built.** The capped program of 16.6 is built per scorer: variables are the 100 best hop-1 candidates (hop-1 nodes of A(q) not in the F-list, by that scorer, ties to the lower node id) and, for each parent in those 100 and each hop-1 node that is in the F-list, its 50 best stored out-neighbours not in the F-list and not in A(q). **A hop-2 candidate's closure lists every parent of that scorer's parent set that has a stored edge to it, not only the parent that proposed it; a parent in the F-list makes it free.** A candidate's value is the maximum over its stored edges and over the parents that propose it. w = value minus the smallest value among that scorer's candidates. Self-test: all eight scorers' candidates, values, weights and closures equal plain loops over explicit unit vectors on 48 toy questions; HiGHS equals exhaustive search on 762 small programs; the worker processes return what an in-process solve returns; the rule of 16.8 on 10 stand-in sets of calls.
- **(c) W5 and the derangement, as built.** W5 is a fixed 64-bit mix of (node, question), read as a number in [0, 1): the same order wherever a node is met (an i.i.d. U(0, 1) per node and question). The derangement is a fixed random cyclic derangement of the population (numpy default_rng(9)); a question never meets its own vector. R1's rotation operators (G-R) and x_q = P q / |P q| from the question's raw vector.
- **(d) The lambda = 0 consequence, found in R2a's records before any Part B number.** R2a's chosen trav lambda is **0.00 on MuSiQue, MetaQA and WebQSP and 0.25 on HotpotQA and 2Wiki** (outputs/sem2/r2/read/e3v/<dataset>.json). **At lambda = 0, W3 and W4 are the same program as W2**: on those three datasets the declared main call W3 - W0 is W2 - W0 (the operator's plausibility without the question, against the question's own cosine), and **the declared control W3 - W4 is exactly zero, LEVEL, so it cannot hold**. 16.8 requires the control to hold wherever the main call is ABOVE and the main call ABOVE on four of the five ROOM datasets; at most two datasets (HotpotQA, 2Wiki) can have a control that holds, so **under the declared rule the verdict cannot be ROUTING_CARRIES: it is ROUTING_NO_QUESTION or NOT_CARRIED by construction.** That is a flaw of the declaration (it read lambda from R2a, where it measured something else); I keep the declared calls and verdict and report them as declared, with this sentence beside them. Identical programs are solved once and the result reused; the report counts them.
- **(e) Two beside scorers, fixed now, no fit.** **W6 = plaus + 1.0 x_q . x_u** and **W7 = plaus + 1.0 x_q' . x_u** (the derangement's question). One fixed number: both terms are cosines on the same sphere, so unit weight is the scale-free choice; nothing is searched. **Beside calls (decide nothing): W6 - W0, W6 - W7 (the control at unit lambda), W6 - W2, W2 - W0**, each at every budget of the ladder, with the paired interval. **Beside verdict at unit lambda:** the rule of 16.8 with W6 - W0 as the main call and W6 - W7 as the control. It is **not** the verdict. If the user wants the question read at unit weight to decide, that is a new declaration on the user's word; it is named here as a candidate, not declared.
- **(f) Gates added.** **G-B:** the population's ids are Part A's, and for every population question the hop-1 size |A| and the missed-gold count recomputed here equal Part A's npz (all of them); base5 equals Part A's. **G-C:** the adjacency table's neighbours equal a direct scan of the graph's keys (Part A's second code path) on 40 random nodes and the seeds of the first three population questions. **G-R** as declared (R1's operators, `--threads 8`). **G-P (Part B):** at least 99.5 percent of the instances proven optimal at 60 s and every solution satisfying the closure and the budget; **a contrast drops the questions where either arm is not proven at that budget** (counted and listed), never reads them as optimal.
- **(g) Rows.** A code check first, `r8bsmoke-webqsp` and `r8bsmoke-2wiki` (20 questions each, the largest graphs and the heaviest fields; read for timing and gates only, never for a completion rate), then `r8b-<dataset>` on the five ROOM datasets, CPU only, 8 cpus, `--threads 8 --workers 6`, placed before the r6 rows; memory is R7's ask for the dataset plus the adjacency table. **Priority is unchanged: pools and links first.**
- **(h) Expectations written now, before any Part B number (added to XB1 and XB2, which stand).**
  - **XB3.** On MuSiQue, MetaQA and WebQSP **W3 - W4 is exactly 0.00** (the same program) and **W2 - W0 is BELOW by at least 5 points on at least two of the three**: the question's cosine is a stronger signal than an operator's plausibility from a parent. So the declared main call there is BELOW and the declared verdict is NOT_CARRIED.
  - **XB4.** At unit lambda **W6 - W0 is ABOVE on at least three of the five ROOM datasets, by at most 10 points** (the operator breaks ties among question-similar neighbours), LEVEL or BELOW on WebQSP, where the hop-1 field is 14,862 nodes and the cap of 100 loses the gold; **W6 - W7 is ABOVE on all five** (the question is read). I do not know whether the beside verdict is ROUTING_CARRIES or NOT_CARRIED; I expect NOT_CARRIED.
  - **XB5.** The candidate recall of W0 (every missed gold inside the capped set) is lowest on WebQSP and below 50 percent there, and below 90 percent on the others; completion at 1,500 is bounded by it.
  - **XB6.** G-P holds (99.5 percent proven optimal at 60 s). If it does not, the instances are dropped as declared and counted, and the call is read on the rest.

### 16.14 Part B filed: NOT_CARRIED (26 November 2026, 14:31 IST, 09:01 UTC)

- **Records.** `outputs/sem2/r8/<dataset>/route.json` and `.npz` for the five ROOM datasets, `report_b.json`, `report_b.md`; the json and md are committed, the npz are not (16.10). Code `outputs/mp_unified/zsem2r8b.py` (sha256 8ddf912f3529..., committed c6816eca, its adjacency build rewritten in a044293a before any Part B number) and `outputs/mp_unified/zsem2r8w.py` (sha256 5b411e9181f8..., c6816eca); **all five records carry those two shas and none is a smoke run**; `report` refuses a record whose sha differs. The report was written on the laptop from the five records; **no instance was dropped on any dataset**.
- **Gates, all five: G-E, G-R (`--threads 8`), G-Q, G-C, G-B, G-P PASS.** G-P: every instance proven optimal at the 60-second limit, every solution inside the closure and the budget (counts below). An instance is one scorer's program at one budget of the ladder for one question; where two scorers have the identical program (W3 and W4 are W2 when lambda is 0.00, 16.13(d)) it was solved once and reused.

| dataset | population of the completable | instances solved | identical programs reused | proven optimal | row time | peak memory (ask) |
|---|---|---:|---:|---|---:|---|
| MuSiQue | 503 of 503 | 12,072 | 4,024 | 12,072 (100.0 percent) | 22 min | 4.20 GB (ask 12) |
| MetaQA | 600 of 994 | 14,400 | 4,800 | 14,400 (100.0 percent) | 10 min | 3.05 GB (ask 10) |
| WebQSP | 600 of 1,108 | 14,400 | 4,800 | 14,400 (100.0 percent) | 122 min | 18.34 GB (ask 28) |
| HotpotQA | 242 of 242 | 7,744 | 0 | 7,744 (100.0 percent) | 29 min | 17.33 GB (ask 26) |
| 2Wiki | 600 of 782 | 19,200 | 0 | 19,200 (100.0 percent) | 68 min | 20.57 GB (ask 27) |

- **Rows, and the events around them, none of which touches a rule.** (1) The memory asks were 12, 10, 28, 26 and 27 GB (MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki); **the measured peaks were lower on every row** (table), and no ask was raised (the host's memory is shared with the link rows, which come first). (2) The five rows sit in the feed **before the stuck head `gp1xr-2wiki`** (hostfeed practice: a head that cannot start holds everything behind it); `r8b-2wiki` was itself the head holding the later rows at 13:03 IST and was sent at 13:08. (3) The first WebQSP smoke (20 questions, read for timing and gates only: all six gates PASS, 480 of 480 programs proven optimal, about 14 s a question) **peaked at 37.7 GB against its 20 GB ask**, because the adjacency table has 1,011,100,859 rows and the first build sorted them in one piece; the build was rewritten as two passes (count, place) with a block-wise sort and committed as a044293a **before any Part B number**, with the self-test comparing it to the one-sort construction at four chunk and block sizes. The re-smoke row starved behind the link rows for 55 minutes and was withdrawn, unsent: the rewrite is checked at scale by G-C and by the full WebQSP row's own placed-row assertion, and the full WebQSP row ran at a peak of 18.34 GB. (4) The smoke records sit under `outputs/sem2/r8/smoke/` and are read by nothing. **16.13(g) said the rows sit before the r6 rows; they were queued there and later moved to just before the stuck head (2).**
- **The completion rates** (percent of the population completable at M' = 1,500; the population is a seeded draw of at most 600; lambda is R2a's trav lambda for that dataset). W7 is W6 with another question; W4 is W3 with another question.

| dataset | population | lambda | W0 qcos | W1 lcos | W2 plaus | W3 routed | W4 other q | W5 random | W6 | W7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue | 503 | 0.00 | 30.0 | 31.6 | 21.3 | 21.3 | 21.3 | 9.7 | 30.6 | 21.7 |
| MetaQA | 600 | 0.00 | 44.7 | 26.8 | 37.3 | 37.3 | 37.3 | 21.5 | 43.2 | 29.8 |
| WebQSP | 600 | 0.00 | 32.2 | 22.5 | 26.3 | 26.3 | 26.3 | 1.5 | 36.7 | 32.3 |
| HotpotQA | 242 | 0.25 | 60.7 | 50.0 | 46.7 | 50.8 | 43.8 | 16.1 | 57.9 | 39.3 |
| 2Wiki | 600 | 0.25 | 14.2 | 35.7 | 29.7 | 28.7 | 30.0 | 6.5 | 26.7 | 27.0 |

- **The declared calls (16.8), with the size of each.** Main = W3 - W0 at M' = 1,500, in points of completion, with the paired bootstrap interval (2,000 resamples, seed 0); control = W3 - W4; beside, at unit lambda (16.13(e)), W6 - W0 and W6 - W7.

| dataset | main W3 - W0 [95%] | call | control W3 - W4 [95%] | call | beside W6 - W0 [95%] | call | beside W6 - W7 [95%] | call |
|---|---|---|---|---|---|---|---|---|
| MuSiQue | -8.75 [-13.52, -3.98] | BELOW | +0.00 [+0.00, +0.00] | LEVEL | +0.60 [-3.38, +4.17] | LEVEL | +8.95 [+5.37, +12.52] | ABOVE |
| MetaQA | -7.33 [-11.67, -3.17] | BELOW | +0.00 [+0.00, +0.00] | LEVEL | -1.50 [-5.17, +2.00] | LEVEL | +13.33 [+10.33, +16.33] | ABOVE |
| WebQSP | -5.83 [-10.00, -1.83] | BELOW | +0.00 [+0.00, +0.00] | LEVEL | +4.50 [+0.83, +8.17] | ABOVE | +4.33 [+1.33, +7.17] | ABOVE |
| HotpotQA | -9.92 [-17.36, -2.48] | BELOW | +7.02 [+3.31, +10.74] | ABOVE | -2.89 [-9.09, +3.31] | LEVEL | +18.60 [+12.40, +24.79] | ABOVE |
| 2Wiki | +14.50 [+10.67, +18.33] | ABOVE | -1.33 [-3.17, +0.17] | LEVEL | +12.50 [+9.17, +15.83] | ABOVE | -0.33 [-3.33, +2.83] | LEVEL |

- **Verdict: NOT_CARRIED.** The rule of 16.8 needs the main call ABOVE on at least two thirds of the five (four) and BELOW on none. It is **ABOVE on one (2Wiki, +14.50 points, interval +10.67 to +18.33) and BELOW on four** (MuSiQue -8.75, MetaQA -7.33, WebQSP -5.83, HotpotQA -9.92). The control holds on one dataset only (HotpotQA, +7.02); on the one dataset where the router beats the question's cosine, 2Wiki, **the control is LEVEL (-1.33, interval -3.17 to +0.17): that gain is the stored label's, not the question's.** On three datasets the control is 0.00 by construction (16.13(d)). **The beside verdict at unit lambda is NOT_CARRIED too:** W6 - W0 is ABOVE on two of five (WebQSP +4.50, 2Wiki +12.50), LEVEL on three, BELOW on none, and W6 - W7 is ABOVE on four. Nothing in 16.11's ROUTING_CARRIES branch follows: **no full-carve (s1eval) read is declared, no pool, link or PL change follows, nothing is trained.**
- **The same contrasts at every budget** (points of completion; A = ABOVE, L = LEVEL, B = BELOW; beside readings except the 1,500 column of the first row of each dataset, which is the declared main call):

| dataset | contrast | M' = 25 | 100 | 500 | 1,500 |
|---|---|---:|---:|---:|---:|
| MuSiQue | W3 - W0 | -1.79 B | -3.38 B | -9.15 B | -8.75 B |
| MuSiQue | W6 - W0 | +0.00 L | +0.00 L | -1.59 L | +0.60 L |
| MuSiQue | W6 - W7 | +1.79 A | +3.78 A | +9.94 A | +8.95 A |
| MuSiQue | W0 - W5 | +2.39 A | +4.97 A | +16.70 A | +20.28 A |
| MuSiQue | W2 - W0 | -1.79 B | -3.38 B | -9.15 B | -8.75 B |
| MetaQA | W3 - W0 | -0.33 L | +2.17 A | +5.50 A | -7.33 B |
| MetaQA | W6 - W0 | +3.33 A | +7.83 A | +11.17 A | -1.50 L |
| MetaQA | W6 - W7 | +2.67 A | +6.83 A | +12.00 A | +13.33 A |
| MetaQA | W0 - W5 | +0.67 L | +1.83 A | +6.83 A | +23.17 A |
| MetaQA | W2 - W0 | -0.33 L | +2.17 A | +5.50 A | -7.33 B |
| WebQSP | W3 - W0 | -1.17 L | -0.67 L | -1.33 L | -5.83 B |
| WebQSP | W6 - W0 | +1.50 L | +6.00 A | +10.33 A | +4.50 A |
| WebQSP | W6 - W7 | +1.00 L | +3.33 A | +6.00 A | +4.33 A |
| WebQSP | W0 - W5 | +3.67 A | +7.67 A | +19.50 A | +30.67 A |
| WebQSP | W2 - W0 | -1.17 L | -0.67 L | -1.33 L | -5.83 B |
| HotpotQA | W3 - W0 | +2.48 L | +0.41 L | -11.16 B | -9.92 B |
| HotpotQA | W6 - W0 | +3.72 L | +1.65 L | -4.13 L | -2.89 L |
| HotpotQA | W6 - W7 | +4.55 A | +7.02 A | +18.60 A | +18.60 A |
| HotpotQA | W0 - W5 | +4.13 A | +13.22 A | +47.52 A | +44.63 A |
| HotpotQA | W2 - W0 | +1.65 L | -0.83 L | -18.18 B | -14.05 B |
| 2Wiki | W3 - W0 | +1.33 A | +4.17 A | +13.00 A | +14.50 A |
| 2Wiki | W6 - W0 | +1.33 A | +4.17 A | +10.33 A | +12.50 A |
| 2Wiki | W6 - W7 | +0.67 L | +2.00 L | +2.33 L | -0.33 L |
| 2Wiki | W0 - W5 | +0.33 L | +3.00 A | +10.00 A | +7.67 A |
| 2Wiki | W2 - W0 | +1.17 A | +3.83 A | +11.00 A | +15.50 A |

  Two things in it are not at the main budget and so decide nothing: on MetaQA, W3 - W0 is ABOVE at 100 and 500 added nodes (+2.17, +5.50) and BELOW only at 1,500 (-7.33), and W6 - W0 is ABOVE at 25, 100 and 500 (+3.33, +7.83, +11.17) and LEVEL at 1,500 (-1.50); on WebQSP and 2Wiki W6 - W0 is ABOVE at 500 (+10.33 each). At a small budget the operator beside the question helps; at 1,500 the question's cosine alone has caught up on MetaQA.
- **Candidate recall, recovered share, branches and cost** (percent; branches opened and cost used are W3's, the declared router; candidate recall is the share of the population whose every missed gold lies inside the capped candidate set of that scorer, 16.6; recovered share is the share of all missed golds inside the solution):

| dataset | cand. recall W0 | W1 | W3 | W6 | W5 | recovered W0 | W3 | W6 | branches median / p90 | cost used median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| MuSiQue | 32.6 | 34.2 | 23.7 | 32.2 | 14.7 | 31.2 | 23.5 | 32.2 | 100 / 100 | 1,500.0 |
| MetaQA | 62.7 | 53.3 | 47.7 | 56.7 | 44.5 | 47.9 | 45.5 | 48.9 | 100 / 100 | 1,500.0 |
| WebQSP | 36.2 | 35.8 | 27.7 | 38.7 | 3.8 | 36.3 | 27.8 | 44.4 | 100 / 100 | 1,273.5 |
| HotpotQA | 61.6 | 52.1 | 52.1 | 59.1 | 22.7 | 60.9 | 51.0 | 58.0 | 100 / 100 | 1,500.0 |
| 2Wiki | 14.3 | 35.8 | 28.8 | 26.8 | 10.0 | 16.1 | 31.6 | 28.7 | 100 / 100 | 1,500.0 |

- **What the room-closed rate is in points of coverage** (implied gain = completion rate x H(1,500) of 16.12, **not a pool gain**: 16.7 says R8's pool is the F-list plus the program's nodes, not PL's cells, and no number of R8 replaces PL's):

| dataset | H(1,500) | W0 | W1 | W3 | W6 | best arm by completion |
|---|---:|---:|---:|---:|---:|---|
| MuSiQue | 32.79 | 9.84 | 10.36 | 6.97 | 10.04 | W1 31.6 percent (10.36 points) |
| MetaQA | 66.40 | 29.66 | 17.82 | 24.79 | 28.66 | W0 44.7 percent (29.66 points) |
| WebQSP | 73.52 | 23.65 | 16.54 | 19.36 | 26.96 | W6 36.7 percent (26.96 points) |
| HotpotQA | 16.05 | 9.75 | 8.03 | 8.16 | 9.28 | W0 60.7 percent (9.75 points) |
| 2Wiki | 52.27 | 7.41 | 18.64 | 14.99 | 13.94 | W1 35.7 percent (18.64 points) |

- **By the number of missed golds** (completion at 1,500, percent): the question needs more than one gold to be found and the rate falls.

| dataset | missed golds | questions | W0 | W3 | W6 | W5 |
|---|---|---:|---:|---:|---:|---:|
| MuSiQue | 1 | 437 | 31.8 | 23.3 | 32.3 | 11.0 |
| MuSiQue | 2 | 64 | 18.8 | 7.8 | 20.3 | 1.6 |
| MuSiQue | 3+ | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| MetaQA | 1 | 252 | 67.5 | 53.6 | 63.1 | 40.5 |
| MetaQA | 2 | 100 | 48.0 | 33.0 | 42.0 | 16.0 |
| MetaQA | 3+ | 248 | 20.2 | 22.6 | 23.4 | 4.4 |
| WebQSP | 1 | 313 | 46.6 | 37.7 | 53.4 | 2.9 |
| WebQSP | 2 | 93 | 29.0 | 24.7 | 31.2 | 0.0 |
| WebQSP | 3+ | 194 | 10.3 | 8.8 | 12.4 | 0.0 |
| HotpotQA | 1 | 241 | 61.0 | 51.0 | 58.1 | 16.2 |
| HotpotQA | 2 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2Wiki | 1 | 429 | 19.1 | 38.0 | 36.1 | 8.6 |
| 2Wiki | 2 | 171 | 1.8 | 5.3 | 2.9 | 1.2 |

- **Expectations, scored against the numbers** (16.9 and 16.13(h); a failed one is reported, not explained away).
  - **XB1 FAILED in three of its four parts.** (i) *Random under 5 percent:* **FAILED**, W5 completes 9.7 (MuSiQue), 21.5 (MetaQA), 1.5 (WebQSP), 16.1 (HotpotQA) and 6.5 percent (2Wiki): only WebQSP is under 5. The capped candidate set (100 branches) is small beside a field of a few hundred (MetaQA's median is 229), so a random pick of 100 completes one question in five there. (ii) *W0 completes 20 to 60 percent:* **FAILED on two of five** (30.0, 44.7, 32.2 inside; HotpotQA 60.7, 0.7 over; 2Wiki 14.2, under). (iii) *W3 - W0 LEVEL or at most +5:* **FAILED**, BELOW on four (-5.83 to -9.92) and +14.50 on 2Wiki. (iv) *NOT_CARRIED or ROUTING_NO_QUESTION:* **held** (NOT_CARRIED). *The control holds:* **FAILED**, it holds on HotpotQA alone (16.13(d) had already shown it cannot on three).
  - **XB2 FAILED.** I expected a median under 20 branches and a long tail. The median is **100 on every dataset, p90 100**: the capped candidate set. The scored program is a knapsack of non-negative weights, so it spends the budget (cost used median 1,500 on every dataset but WebQSP, 1,273.5) and opens branches until the cap or the budget stops it; the number reads how the program fills, not what the question needs. What the question needs is Part A's cost*: a median of 1 to 2 added nodes. **The declared measure could not have shown a thin path; I read it as if it could.**
  - **XB3 held.** W3 - W4 is exactly +0.00 on MuSiQue, MetaQA and WebQSP; **W2 - W0 is BELOW on all three** (-8.75, -7.33, -5.83: at least 5 points on at least two was expected); the declared main call is BELOW on those three and the declared verdict is NOT_CARRIED.
  - **XB4 FAILED in two of three parts.** At unit lambda, W6 - W0 ABOVE on at least three of five: **FAILED**, two (WebQSP +4.50, 2Wiki +12.50, which is over the 10-point ceiling written); WebQSP was expected LEVEL or BELOW and is ABOVE. W6 - W7 ABOVE on all five: **FAILED on 2Wiki** (-0.33, LEVEL), ABOVE on four (+4.33 to +18.60). The beside verdict **NOT_CARRIED held**.
  - **XB5 FAILED in one part.** *Lowest on WebQSP:* **FAILED**, W0's candidate recall is lowest on **2Wiki (14.3)**, then MuSiQue 32.6, WebQSP 36.2, HotpotQA 61.6, MetaQA 62.7. *Below 50 percent on WebQSP:* held (36.2). *Below 90 on the others:* held (all under 63). *Completion bounded by it:* held on every dataset (W0 completes 30.0 of 32.6, 44.7 of 62.7, 32.2 of 36.2, 60.7 of 61.6 and 14.2 of 14.3).
  - **XB6 held.** Every instance proven optimal on all five (100.0 percent against 99.5 declared); none dropped.
- **What the numbers say and what they do not.**
  - **(1) The question's own cosine beats the stored label's plausibility on four of five; the label alone is not a route.** W2 (the operator of the stored label applied to the parent, no question) is BELOW W0 by 8.75, 7.33, 5.83 and 14.05 points on MuSiQue, MetaQA, WebQSP and HotpotQA and ABOVE it by 15.50 only on 2Wiki, where W0 is weakest (14.2 percent, against 35.7 for the plain along-link cosine W1): there the parent's geometry carries what the question's does not.
  - **(2) The question is read, and it is a strong signal against chance.** W0 - W5 is ABOVE on all five (+20.28, +23.17, +30.67, +44.63, +7.67 points); at unit lambda W6 - W7 is ABOVE on four (+8.95, +13.33, +4.33, +18.60). At the declared lambda, W3 - W4 is ABOVE on HotpotQA (+7.02) and LEVEL on 2Wiki (-1.33).
  - **(3) The operator beside the question adds little at the main budget.** W6 - W0 is LEVEL on MuSiQue, MetaQA and HotpotQA, ABOVE on WebQSP (+4.50) and 2Wiki (+12.50) (above).
  - **(4) The room closed is at most 61 percent, and the cap bounds it.** The best completion on any dataset is 60.7 (HotpotQA, W0), 44.7 (MetaQA, W0), 36.7 (WebQSP, W6), 35.7 (2Wiki, W1) and 31.6 (MuSiQue, W1): none above two thirds of a population where a perfect router completes all (H, 16.12). In points of coverage that is 9.75, 29.66, 26.96, 18.64 and 10.36 of the 16.05, 66.40, 73.52, 52.27 and 32.79 on offer; **not a pool gain** (16.7). **On MuSiQue, HotpotQA and 2Wiki completion is 92 to 99 percent of the candidate recall: the loss is the cap (the gold, or its parent, is not among the scorer's best 100 hop-1 nodes and the best 50 out-neighbours of each), not the budget.** On MetaQA the budget costs 29 percent of the recall (44.7 of 62.7), on WebQSP 11 percent (32.2 of 36.2).
  - **(5) A gold's parent is the cost.** Completion falls with the number of missed golds (MetaQA W0: 67.5, 48.0, 20.2 percent for one, two and three or more).
  - **(6) A caveat on the caps.** The caps (100 and 50, 16.6) are declared and not varied; a larger cap would raise the candidate recall at a larger program. R8 does not say where the recall tops out.

### 16.15 Beside: the question-conditioned semantic b-tree (the user's picture, 26 November 2026; decides nothing, no new number)

The user's picture, in their words: "a question depending semantic b-tree": one stored structure over the nodes, the question choosing the branch at each step, as a descent through a tree of separators, so that the walk touches a thin path and not the field. This section says what R8's two parts show about it, and what they cannot.

- **What Part A would show if the picture held, and what it does show.** If a thin, question-chosen path exists, the oracle's walk should be tiny, far below the unordered walk and the budget, and its cost should not grow with the field. **It is:** cost* is 1 to 2 added nodes (median), 1.01 to 1.17 per missed gold; the budget does not bind (H(25) = H(1,500) on MuSiQue, HotpotQA and 2Wiki); the field is 229 to 14,862 hop-1 nodes and the unordered cost 10^4 to 10^8. **That is the necessary condition, and it holds.** Had it failed (large cost*, H rising with the budget, golds mostly beyond two hops), the picture would have been dead on arrival. It says a thin path exists; it does not say the question can find it.
- **What Part B shows about choosing the branch.** The nearest thing R8 has to one level of separators is the flat cosine of the question to the hop-1 field. It chooses far better than chance (W0 - W5, +7.67 to +44.63 points) and better than another question (W6 - W7 ABOVE on four), so the question is read; and the stored label's operator is **not** the separator on four of five (W2 below W0). But it finds the thin path on **at most 61 percent of the questions** with 100 branches and 1,500 added nodes (14 to 61 percent across the five), and its candidate set holds every missed gold on only 14 to 63 percent. So the picture is half supported: **the question does choose, and a thin path exists, but one flat question-to-node cosine does not choose well enough** on four of five, and the limit is the candidate set, not the budget.
- **What Part B cannot say.** It never builds a tree: its scorers rank the nodes of the field by one flat score, whose leaves are nodes. A tree would add **coarse separators** (the centroids of groups of nodes, which average the noise of single node vectors) and descend: choose the group, then the branch inside it. Whether a coarse-to-fine descent holds the gold where the flat cosine does not is **not measured**; R8 fits nothing and builds no index (16.7).
- **A candidate round, named and not declared (a new declaration on the user's word only): a descent router.** Build a label-free hierarchical index over the node vectors (a balanced tree by recursive k-means; the branching factor and the leaf size set by the budget, not fitted: a leaf is at most as large as a branch the budget can pay), once per dataset, and let the question descend by comparing its vector with the separators (the centroids) with a beam of width w; the pool adds the leaves the descent reaches that the stored links connect to the seeds. **Read:** the share of questions whose every missed gold lies in the reached leaves (a tree recall, set beside W0's candidate recall at the same number of added nodes), the completion by the same population and program, and Part A's cost* as the ceiling. **Controls, declared with it:** a tree of the same shape with random assignment, another question's descent, and the flat W0 at the same node count. **A learned separator or adapter** is a further step (16.7's trained router) and needs its own word. CPU only; it reuses `zsem2r8b.py`'s adjacency table and worker module, and nothing is fitted on the questions. **Priority is unchanged: pools and links first; no pool, link or PL change follows from R8.**

## 17. R9 declared: the cap ladder and the descent router (26 November 2026, 14:45 IST, 09:15 UTC), before any R9 number

### 17.1 Naming, why, and what was seen

- **Naming.** The user's picture, 26 November: "a question depending semantic b-tree" (16.15), asked for twice as "try testing this"; the session that started this one passed it on after R8's Part B was filed (8eee96d0) and asked for the descent router named in 16.15, preceded by the cap ladder. **This page is R9**, the round after R8. It is declared on that word, CPU only, on a dev slice first (no s1eval question is read), with a new module `outputs/mp_unified/zsem2r9.py` that imports `zsem2r8b.py`, `zsem2r8w.py`, `zsem2r8.py` and the older modules read-only and edits none of them.
- **Why the ladder comes first.** R8's Part B found that on MuSiQue, HotpotQA and 2Wiki the loss is the cap (100 hop-1 nodes, 50 out-neighbours of each), not the budget: completion was 92 to 99 percent of the candidate recall at the cap. A tree can only matter if the flat cosine tops out below what a larger cap reaches, so **Part A9 reads where the flat cosine tops out**, with the same program, before Part B9 asks whether a tree chooses better.
- **Seen before this page:** R8's Part A and Part B records (outputs/sem2/r8), no R9 number. **Not seen:** any candidate recall or completion at a cap other than 100 / 50, any tree. No R9 number exists.

### 17.2 The question, in two parts

- **Part A9 (the cap ladder, no tree).** For the question's own cosine (W0) and the cosine along the stored link (W1) of 16.6, at the same budget M' = 1,500 and the same program: how do the candidate recall and the completion rise as the cap grows from 100 / 50 to 300 / 150 and 1,000 / 500, and what does the hop-1 field's rank of the missed golds say about an uncapped cap?
- **Part B9 (the descent router).** Does a tree of separators over the node vectors, descended by the question, choose the branches of the stored walk better than the node-level cosine at the same number of branches, with the same program?

### 17.3 Population and the dev slice (fixed now)

- **Population = R8's** (`outputs/sem2/r8/<dataset>/route.npz`: ids and `pop`, a seeded draw of at most 600 of the questions completable at 1,500; its record's sha is checked). **Datasets: the five ROOM datasets** (MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki). First stage, seeds, walk, pool and cost as 16.3.
- **Dev slice = every eighth question of R8's population** (positions 0, 8, 16, ... in its order): **MuSiQue 63 of 503, MetaQA 75 of 600, WebQSP 75 of 600, HotpotQA 31 of 242, 2Wiki 75 of 600; 319 questions.** No s1eval question is read; nothing is fitted. **Power, said now:** with 31 to 75 questions a paired interval on a completion difference is wide (R8's intervals on 242 to 600 questions were about plus or minus 4 to 7 points; scaled to 31 to 75 questions they are about plus or minus 11 to 20), so only large effects can be called ABOVE or BELOW; a difference inside the interval is LEVEL and is not evidence of equality. A full-population read is declared only by 17.9.

### 17.4 Part A9: the cap ladder (fixed now)

- **Scorers and candidate sets.** W0 (the question's cosine, parent-blind) and W1 (the cosine along the stored link) of 16.6, with the candidate sets of 16.13(b) at a cap pair **(A_cap, D_cap)**: the A_cap best hop-1 nodes of A(q) minus the F-list by that scorer (ties to the lower id), and for each of them and each hop-1 node that is in the F-list its D_cap best stored out-neighbours not in the F-list and not in A(q). **Rungs: C1 = (100, 50) (R8's), C2 = (300, 150), C3 = (1,000, 500).**
- **Program.** The same Pi_score as 16.4 and 16.13(b): weight w_u = value minus the smallest value among the candidates, closure, budget M' = 1,500, HiGHS branch and bound, gap 0. **Time limit 60 s at C1 and 120 s at C2 and C3** (the programs are 9 to 100 times larger; this is a departure from 16.4's 60 s for those two rungs and it changes no rule). **G-P rule per rung:** a rung's completion is read only if at least 99.5 percent of its instances are proven optimal; unproven instances are dropped from that rung's contrasts, counted and listed; the candidate recall (no solver) is always read.
- **Uncapped, read without building it.** A cap of "all" would make the hop-2 set 10^7 to 10^8 nodes on WebQSP and 2Wiki (16.12's unordered cost), so it is not built. Two reads stand in: **(i)** for every question, **the hop-1 need of W** = the largest rank, in W's order over the field A(q) minus the F-list, of the question's missed golds that lie in the field; reported as the share of questions with need at most 100, 300 and 1,000, and the median and 90th percentile of need and of need / field size, over the questions with at least one missed hop-1 gold (this is exact for the hop-1 golds); **(ii)** the candidate recall of a cap that holds the whole population is 1 by the definition of the population (every missed gold is within two hops), which is a definition and not a measurement.
- **Measures at each rung and scorer:** candidate recall (every missed gold inside the capped set); completion (every missed gold in the program's solution); the recovered share of missed golds; the median candidate count; the branches opened (hop-1 nodes in the solution) and the cost used.
- **No call and no verdict.** Part A9 is a read, as Part A of R8 was a read. Two named readings, decided now: **CAP_BINDS (W)** iff the candidate recall of W at C3 exceeds that at C1 by at least 10 points on at least three of the five datasets; **COMPLETION_FOLLOWS (W)** iff the completion of W at C3 exceeds that at C1 by at least 5 points on at least three of the five datasets (where the rung is read). Beside: the flat ceiling = the best completion of W0 or W1 over the three rungs.
- **Gate G-X.** C1 must reproduce R8: for W0 and W1 the candidate-recall flag equals R8's `cand_recall` for the same question on at least 98 percent of the slice, and the completion equals R8's `comp` at M' = 1,500 on at least 98 percent of the instances proven in both; mismatches are listed (a float tie at the cap boundary is the only expected cause: the new builder scores two columns where R8's scored the operators too).

### 17.5 Part B9: the descent router (fixed now)

- **The index.** One tree per dataset over all of its nodes' vectors x_u = P v_u / |P v_u| (R1's P as stored, gate G-R), **built without labels, questions or fitting**: recursive **spherical k-means with branching b = 16 and leaf size at most L = 100** (b and L are the same on every dataset; L is R8's number of branches). A cluster of more than L members is split by 8 Lloyd steps of spherical k-means fitted on at most 20,000 of its members (init: b distinct random members, numpy default_rng(17 + the cluster's creation number); an empty cluster is re-seeded with a random member), then every member is assigned to its nearest centroid; a split that leaves all members in one child is replaced by b equal chunks in node-id order. **The separators are the unit centroids.** Leaves are numbered in depth-first order, so every vertex covers a range of leaves.
- **Walk-aware descent (the question chooses the branch at each step).** For a question vector x_q and a set S of walk nodes (S1 = A(q) minus the F-list at hop 1; S2 = the stored out-neighbours of the chosen parents, minus the F-list and A(q), at hop 2), the count of a vertex is the number of nodes of S in its subtree (prefix sums over the leaf counts). **Beam search of width w:** keep the best w vertices by cosine of x_q to the vertex's centroid among vertices of count at least 1; expand the non-leaf vertices of the beam into their children (a leaf stays); repeat until the beam holds leaves only. The reached leaves, ordered by cosine, give the arm's candidates: the nodes of S in the leaves, taking leaves in that order while the running count stays within **1,000 at hop 1 and 5,000 at hop 2** (always at least the first leaf; these are the program-size bounds of R8's 100 and 5,100).
- **The router arm T(w).** Hop 1: descend with S1, candidates = chosen hop-1 nodes. Hop 2: parents = the chosen hop-1 nodes plus the hop-1 nodes that are in the F-list; descend again with S2; candidates = the chosen hop-2 nodes. **Beam ladder w in {8, 32, 128, 512}; the main rung is w = 128.** Program: as 17.4 (weights W0's value x_q . x_u, the closure of 16.13(b), M' = 1,500, 60 s).
- **The flat arm F(w), at the same node count.** For each question and rung, with n1 and n2 the number of hop-1 and hop-2 candidates T(w) chose: hop 1: the n1 best nodes of S1 by W0's value; parents = those nodes plus the hop-1 nodes in the F-list; S2 = their stored out-neighbours minus the F-list and A(q); hop 2: the n2 best nodes of S2 by W0's value. Same program. **F is the node-level cosine given exactly the branches and the nodes the tree was given.**
- **Controls, declared with the call.** **R(w):** the same tree with its nodes' leaf assignment randomly permuted (leaf sizes unchanged, numpy default_rng(18)) and every separator recomputed as the unit mean of its random members; the same descent and program (the null of a tree with no semantic structure). **Tq(w):** the router arm descended with the question of a fixed random derangement of the slice (default_rng(9)), evaluated on the question's own golds (the question's contribution).
- **Measures at each rung and arm:** candidate recall; completion at M' = 1,500; the recovered share; n1 and n2 (median, 90th percentile); the branches opened and the cost used.

### 17.6 Calls and the verdict (fixed now)

- **Main call per dataset (w = 128): T - F, completion rate in points over the slice, paired bootstrap over questions (2,000 resamples, seed 0, zsem2.paired):** ABOVE, LEVEL, BELOW. **Controls, per dataset: T - R and T - Tq** (the tree has structure; the question is read): a control holds iff ABOVE.
- **Verdict (the rule of 16.8 with the tree's calls).** **TREE_CARRIES** iff the main call is ABOVE on at least two thirds of the five datasets and BELOW on none, and both controls hold on every dataset where the main call is ABOVE. **TREE_NO_QUESTION** iff the main call carries and a control does not. **NOT_CARRIED** otherwise. **INCONCLUSIVE** if fewer than four datasets could be read (a gate failed or the programs were not proven).
- **Beside (decide nothing):** T - F, T - R, T - Tq at the other rungs; T against R8's flat C1 (100 / 50) completion on the same questions; the candidate recall of each arm at equal counts; the share of the slice whose missed golds all lie in the reached leaves.

### 17.7 Expectations, written before any number (a failed one is reported, not explained away)

- **XC1.** W0's candidate recall rises with the cap on every dataset, and **C3 exceeds C1 by at least 15 points on at least four of five** (CAP_BINDS holds for W0); on WebQSP the C3 recall lies between 50 and 75 percent (the field is 14,862 nodes and 1,000 is 7 percent of it).
- **XC2.** The completion does **not** follow the recall: at C3 the program has 100 to 1,000 times more candidates than it has budget, so the golds of lower cosine are crowded out; on at least three datasets the completion gain C3 over C1 is **less than half** of the recall gain. So COMPLETION_FOLLOWS may hold on fewer than three datasets even though CAP_BINDS holds.
- **XC3.** W1 completes at least as much as W0 at every rung on 2Wiki; W0 completes more than W1 at every rung on MetaQA, WebQSP and HotpotQA (as R8 at C1: 44.7 against 26.8, 32.2 against 22.5, 60.7 against 50.0).
- **XC4.** G-P holds at C1 (100 percent) and at C2 (at least 99.5 percent); **at C3 at least one dataset has fewer than 99.5 percent proven at 120 s**, so its C3 completion is not read (its recall is). I do not know which.
- **XC5.** At the main rung **T - F is BELOW or LEVEL on at least four of the five datasets**; it is ABOVE on at most one. The reason for the expectation: the flat arm is the node-level optimum of the same cosine over exactly the nodes the tree reached, and a tree can only gain where the centroid of a group of nodes is a better separator than a node's own vector; I expect that on a KB (MetaQA, WebQSP) at most.
- **XC6.** The tree reads the question and has structure: **T - R is ABOVE on all five, T - Tq is ABOVE on at least four.**
- **XC7. My expectation of the verdict: NOT_CARRIED (about 85 percent); TREE_CARRIES (about 10 percent); TREE_NO_QUESTION (about 5 percent).** An outcome of INCONCLUSIVE is not expected.
- **What each outcome means for the pools.** **Nothing changes in pools, links, PL or any model from R8 or R9 until the user says so.** NOT_CARRIED: the flat node-level cosine already chooses at least as well as a tree of separators over these vectors, so the picture's missing piece is not a better separator over the same vectors; what is left (17.9) is named, not queued. TREE_CARRIES on the dev slice: it is a signal on 319 questions, read at a slice's power, and only the declared full-population read (17.9) can make it a result.

### 17.8 Gates, code, rows and outputs

- **Gates.** G-R (R1's operators reproduced, `--threads 8`; the P of every x), G-Q (base5 equals Part A's), G-B (the slice's ids, |A| and missed count equal R8's route record), G-C (the adjacency table's neighbours equal a direct scan, as 16.13(f)), **G-X** (17.4), **G-P** (17.4, per rung; for Part B9 at least 99.5 percent of the instances proven optimal at 60 s, a contrast drops the questions where either arm is not proven), **G-T** (the tree: every node is in exactly one leaf; every leaf has at most L nodes, except the equal-chunk fallback's; every vertex's leaf range is the union of its children's; the prefix-sum counts equal a brute-force count on 5 random node sets; the random tree has the leaf-size multiset of the tree; every separator has unit norm), **G-D** (the descent: with w equal to the number of leaves it returns every leaf of count at least 1; the toy beam equals an exhaustive search; the counts are those of the set).
- **Self-test (before any row):** a toy graph and toy vectors: the cap-ladder builder equals plain loops over explicit unit vectors at three cap pairs; the tree and the descent equal brute force; the flat arm's n best equal a sort; the program of an arm equals a hand-built closure; the report's rule on stand-in records.
- **Code:** `outputs/mp_unified/zsem2r9.py`: `selftest`, `cap --ds D --host --threads 8 --workers 6 [--smoke]`, `tree --ds D --host --threads 8 --workers 6 [--smoke]`, `report`. **Records carry the shas of zsem2r9.py, zsem2r8b.py, zsem2r8w.py and zsem2r8.py**, and `report` refuses a record whose sha differs or that is a smoke run.
- **Rows:** a code check first (`r9smoke-<dataset>`: 8 slice questions, read for timing and gates only, never for a number), then `r9cap-<dataset>` and `r9tree-<dataset>` for the five datasets, CPU only, 8 cpus, `--threads 8`. **Memory asks from R8's measured peaks (WebQSP 18.3 GB, 2Wiki 20.6, HotpotQA 17.3, MuSiQue 4.2, MetaQA 3.05) with a small margin; no ask above them is raised.** The links come first: a row goes before a stuck head only where host memory is free. **Reading order: Part A9's numbers are read and filed before any Part B9 number is read.**
- **Outputs:** `outputs/sem2/r9/<dataset>/cap.json` and `.npz`, `tree.json` and `.npz` (the index itself `tree_index.npz` is not committed), then `report_cap.json`, `report_cap.md`, `report_tree.json`, `report_tree.md`; json and md are committed, the npz are not.

### 17.9 What follows (fixed now)

- **Part A9 alone** decides nothing beyond its two named readings; it is filed first.
- **NOT_CARRIED or INCONCLUSIVE:** R9 stops. Named, not queued: separators over the stored walk's own field (a local tree per node's neighbour list), a third hop, a learned separator or adapter (16.7), on the user's word.
- **TREE_CARRIES, or T - F with a point estimate of at least +5 points on at least two datasets and BELOW on none:** a **full-population read** (every question of R8's populations, the same code, one run) is declared on this page as an amendment before it is run; nothing is passed on before it. **No s1eval read, no pool, link or PL change, and nothing is trained, either way.**

### 17.10 What this does not test (named, not run)

- A learned separator, a trained router, an adapter (16.7); a third hop; the K store; a tree over the walk's own field built per question or per node; another branching factor or leaf size; another beam ladder; a different weight in the program; the cross-question allocation; s1eval. The tree is global (one index per dataset) and walk-aware (counts from the stored walk); it is **not** a b-tree of the stored graph's own structure.

### 17.11 Part A9 filed (26 November 2026, 18:32 IST, 13:02 UTC): the cap binds, the completion does not follow; 4 of 5 datasets (2Wiki's row is still running)

**Order kept.** Part A9 is read and filed here before any Part B9 number is read (the tree records of MuSiQue and MetaQA are on disk and unread). **Rows finished:** `r9cap-musique`, `r9cap-metaqa`, `r9cap-hotpotqa`, `r9cap-webqsp` (1,999 s, 389 s, 3,224 s and 9,157 s in the process, the load included). **Not finished:** `r9cap-2wiki` (about 340 s per question; it will end after this filing; its cells are added in 17.13 and cannot change the two named readings, 17.11(c)). **Code:** `outputs/mp_unified/zsem2r9.py` (committed 5a05dd6d before the rows, sha 37ed9283f6b9…, records carry it) and the beside reads `outputs/mp_unified/zsem2r9a.py` (new, read-only on the records; it prints the pooled proven shares and the paired reads declared in 17.4 that the report of zsem2r9.py does not). Reports: `outputs/sem2/r9/report_cap.json|md` and `report_cap_beside.json|md`.

**Gates, all 4: G-E, G-R (`--threads 8`), G-Q, G-C, G-B, G-X, G-P PASS.** G-X: at 100/50 the candidate recall equals R8's on every slice question (1.0 for W0 and W1, no mismatching position) and the completion equals R8's on every instance proven in both (1.0). G-P: every solution satisfied the closure and the budget; the proven shares are in table (a), the unproven programs are counted in table (c) and are not read as optimal.

**(a) The ladder, per dataset and scorer** (candidate recall over the slice; completion over the slice where the rung is read, that is where at least 99.5 percent of the scorer's programs at that rung were proven optimal):

| dataset | slice | scorer | candidate recall % at 100/50, 300/150, 1000/500 | completion % at the same rungs (— = not read) | proven share at the same rungs |
|---|---:|---|---|---|---|
| MuSiQue | 63 | W0 | 27.0 / 55.6 / 90.5 | 27.0 / 27.0 / — | 100.0 / 100.0 / 96.8 |
| MuSiQue | 63 | W1 | 34.9 / 60.3 / 92.1 | 33.3 / 25.4 / — | 100.0 / 100.0 / 85.7 |
| MetaQA | 75 | W0 | 57.3 / 96.0 / 100.0 | 42.7 / 25.3 / — | 100.0 / 100.0 / 98.7 |
| MetaQA | 75 | W1 | 49.3 / 94.7 / 98.7 | 21.3 / 12.0 / 12.0 | 100.0 / 100.0 / 100.0 |
| HotpotQA | 31 | W0 | 45.2 / 90.3 / 100.0 | 45.2 / 41.9 / — | 100.0 / 100.0 / 80.7 |
| HotpotQA | 31 | W1 | 41.9 / 93.5 / 96.8 | 41.9 / — / — | 100.0 / 96.8 / 90.3 |
| WebQSP | 75 | W0 | 42.7 / 54.7 / 62.7 | 36.0 / 34.7 / — | 100.0 / 100.0 / 94.7 |
| WebQSP | 75 | W1 | 37.3 / 46.7 / 60.0 | 24.0 / — / — | 100.0 / 97.3 / 85.3 |

**(b) Beside, decided nothing: the completion against the rung 100/50 on the questions both programs were proven on** (a selected subset, counted: the unproven programs are the large ones, so this is not the slice; the interval is zsem2.paired over those questions):

| dataset | scorer | rung against 100/50 | questions proven at both | completion at 100/50 on them % | completion at the rung on them % | difference, points [interval] | call | recall gain, all questions, points |
|---|---|---|---:|---:|---:|---|---|---:|
| MuSiQue | W0 | 300/150 | 63 of 63 | 26.98 | 26.98 | 0.0 [-11.11, 11.11] | LEVEL | 28.57 |
| MuSiQue | W0 | 1000/500 | 61 of 63 | 26.23 | 24.59 | -1.64 [-13.11, 9.84] | LEVEL | 63.49 |
| MuSiQue | W1 | 300/150 | 63 of 63 | 33.33 | 25.4 | -7.94 [-17.46, 0.0] | LEVEL | 25.4 |
| MuSiQue | W1 | 1000/500 | 54 of 63 | 33.33 | 20.37 | -12.96 [-25.93, 0.0] | LEVEL | 57.14 |
| MetaQA | W0 | 300/150 | 75 of 75 | 42.67 | 25.33 | -17.33 [-26.67, -8.0] | BELOW | 38.67 |
| MetaQA | W0 | 1000/500 | 74 of 75 | 43.24 | 25.68 | -17.57 [-27.03, -8.11] | BELOW | 42.67 |
| MetaQA | W1 | 300/150 | 75 of 75 | 21.33 | 12.0 | -9.33 [-17.33, -2.67] | BELOW | 45.33 |
| MetaQA | W1 | 1000/500 | 75 of 75 | 21.33 | 12.0 | -9.33 [-17.33, -2.67] | BELOW | 49.33 |
| HotpotQA | W0 | 300/150 | 31 of 31 | 45.16 | 41.94 | -3.23 [-22.58, 16.13] | LEVEL | 45.16 |
| HotpotQA | W0 | 1000/500 | 25 of 31 | 44.0 | 12.0 | -32.0 [-52.0, -12.0] | BELOW | 54.84 |
| HotpotQA | W1 | 300/150 | 30 of 31 | 43.33 | 33.33 | -10.0 [-30.0, 10.0] | LEVEL | 51.61 |
| HotpotQA | W1 | 1000/500 | 28 of 31 | 39.29 | 17.86 | -21.43 [-39.29, -3.57] | BELOW | 54.84 |
| WebQSP | W0 | 300/150 | 75 of 75 | 36.0 | 34.67 | -1.33 [-8.0, 5.33] | LEVEL | 12.0 |
| WebQSP | W0 | 1000/500 | 71 of 75 | 36.62 | 26.76 | -9.86 [-18.31, -1.41] | BELOW | 20.0 |
| WebQSP | W1 | 300/150 | 73 of 75 | 24.66 | 17.81 | -6.85 [-13.7, 0.0] | LEVEL | 9.33 |
| WebQSP | W1 | 1000/500 | 64 of 75 | 23.44 | 14.06 | -9.38 [-18.75, 0.0] | LEVEL | 22.67 |

**(c) The named readings, decided now.** **CAP_BINDS: TRUE for W0 and for W1** (the candidate recall at 1000/500 over 100/50 is at least 10 points on every dataset read: W0 MuSiQue +63.5, MetaQA +42.7, HotpotQA +54.8, WebQSP +20.0; W1 MuSiQue +57.1, MetaQA +49.3, HotpotQA +54.8, WebQSP +22.7; three of five are needed and 4 are read). **COMPLETION_FOLLOWS: FALSE for W0 and for W1**, and this is decided without 2Wiki: the completion gain at 1000/500 over 100/50 is read only where the rung is read (MetaQA W1: -9.33; no other cell of the four datasets is read at that rung), no read gain reaches 5 points, three of five datasets are needed and one dataset is left, so it cannot reach three. **The flat ceiling, the best completion read over the rungs and scorers, is the 100/50 rung on every dataset read:** MuSiQue 33.3, MetaQA 42.7, HotpotQA 45.2, WebQSP 36.0 (R8's own cap).

**(d) Proven shares, per rung** (the declared text says "a rung's instances", which pooled over both scorers is the stricter reading; zsem2r9.py reads per scorer and rung; the cells where the two differ are in the last column and **no named reading depends on them**):

| dataset | rung | unproven programs | proven share pooled over both scorers | W0 | W1 | cells the pooled reading would not read although the per-scorer reading did |
|---|---|---:|---:|---:|---:|---|
| MuSiQue | 100/50 | 0 | 100.0 | 100.0 | 100.0 | — |
| MuSiQue | 300/150 | 0 | 100.0 | 100.0 | 100.0 | — |
| MuSiQue | 1000/500 | 11 | 91.3 | 96.8 | 85.7 | — |
| MetaQA | 100/50 | 0 | 100.0 | 100.0 | 100.0 | — |
| MetaQA | 300/150 | 0 | 100.0 | 100.0 | 100.0 | — |
| MetaQA | 1000/500 | 1 | 99.3 | 98.7 | 100.0 | W1 |
| HotpotQA | 100/50 | 0 | 100.0 | 100.0 | 100.0 | — |
| HotpotQA | 300/150 | 1 | 98.4 | 100.0 | 96.8 | W0 |
| HotpotQA | 1000/500 | 9 | 85.5 | 80.7 | 90.3 | — |
| WebQSP | 100/50 | 0 | 100.0 | 100.0 | 100.0 | — |
| WebQSP | 300/150 | 2 | 98.7 | 100.0 | 97.3 | W0 |
| WebQSP | 1000/500 | 15 | 90.0 | 94.7 | 85.3 | — |

**(e) The hop-1 need, the stand-in for an uncapped cap** (the largest rank of a question's missed hop-1 golds in the scorer's order over the hop-1 field minus the F-list, over the questions that have such a gold):

| dataset | questions with a missed hop-1 gold | scorer | need at most 100 / 300 / 1,000 (share of those questions, %) | need median, 90th percentile | need over field size, median, 90th percentile |
|---|---:|---|---|---|---|
| MuSiQue | 46 | W0 | 26.1 / 50.0 / 89.1 | 310, 1010 | 0.318, 0.755 |
| MuSiQue | 46 | W1 | 37.0 / 58.7 / 93.5 | 174, 890 | 0.187, 0.651 |
| MetaQA | 47 | W0 | 61.7 / 95.7 / 100.0 | 60, 213 | 0.321, 0.824 |
| MetaQA | 47 | W1 | 51.1 / 97.9 / 100.0 | 97, 247 | 0.528, 0.875 |
| HotpotQA | 30 | W0 | 46.7 / 90.0 / 100.0 | 102, 276 | 0.208, 0.534 |
| HotpotQA | 30 | W1 | 43.3 / 96.7 / 100.0 | 150, 256 | 0.263, 0.572 |
| WebQSP | 70 | W0 | 42.9 / 52.9 / 60.0 | 200, 7337 | 0.027, 0.887 |
| WebQSP | 70 | W1 | 40.0 / 50.0 / 64.3 | 310, 9088 | 0.022, 0.763 |

**(f) W0 minus W1 at each rung, on the questions proven in both** (beside; XC3):

| dataset | rung | questions proven in both | completion W0 minus W1, points [interval] | call | recall W0 minus W1, points |
|---|---|---:|---|---|---:|
| MuSiQue | 100/50 | 63 | -6.35 [-20.63, 7.94] | LEVEL | -7.94 |
| MuSiQue | 300/150 | 63 | 1.59 [-12.7, 15.87] | LEVEL | -4.76 |
| MuSiQue | 1000/500 | 52 | 7.69 [-9.62, 25.0] | LEVEL | -1.59 |
| MetaQA | 100/50 | 75 | 21.33 [8.0, 36.0] | ABOVE | 8.0 |
| MetaQA | 300/150 | 75 | 13.33 [1.33, 26.67] | ABOVE | 1.33 |
| MetaQA | 1000/500 | 74 | 14.86 [2.7, 27.03] | ABOVE | 1.33 |
| HotpotQA | 100/50 | 31 | 3.23 [-16.13, 22.58] | LEVEL | 3.23 |
| HotpotQA | 300/150 | 30 | 6.67 [-13.33, 26.67] | LEVEL | -3.23 |
| HotpotQA | 1000/500 | 24 | -4.17 [-16.67, 8.33] | LEVEL | 3.23 |
| WebQSP | 100/50 | 75 | 12.0 [0.0, 24.0] | LEVEL | 5.33 |
| WebQSP | 300/150 | 73 | 17.81 [5.48, 30.14] | ABOVE | 8.0 |
| WebQSP | 1000/500 | 60 | 11.67 [0.0, 23.33] | LEVEL | 2.67 |

**(g) The expectations of 17.7 that Part A9 can score (a failed one is reported as failed).**
- **XC1 HELD.** W0's candidate recall rises with the cap on every dataset read (27.0, 55.6, 90.5 on MuSiQue; 57.3, 96.0, 100.0 on MetaQA; 45.2, 90.3, 100.0 on HotpotQA; 42.7, 54.7, 62.7 on WebQSP) and the gain of 1000/500 over 100/50 is at least 15 points on all 4 read (so on at least four of five whatever 2Wiki reads); WebQSP's recall at 1000/500 is 62.7, inside 50 to 75.
- **XC2 CANNOT BE SCORED AS DECLARED, and the direction I expected is there beside.** XC2 asks the completion gain of 1000/500 over 100/50 to be less than half of the recall gain on at least three datasets, and that rung's completion is read in one cell of the 8 (MetaQA W1: -9.3 against a recall gain of +49.3, which holds). Beside, on the proven subset of (b), W0's completion at 1000/500 against 100/50 is -1.6 (MuSiQue), -17.6 (MetaQA), -32.0 (HotpotQA) and -9.9 (WebQSP) points against recall gains of +63.5, +42.7, +54.8 and +20.0: below half on all four, BELOW on three; W1 is -13.0, -9.3, -21.4 and -9.4. **At 300/150, where the rung is read for W0 on all four, the completion is at or below 100/50 on all four** (0.0, -17.3, -3.2, -1.3 points) while the recall gains +28.6, +38.7, +45.2, +12.0. The reason I wrote for it (the candidate sets are 7 to 36 times the budget at 1000/500 by their medians, and crowd the golds of lower cosine out) is **not tested** here: Part A9 measures the completion, not the displacement.
- **XC3 FAILED IN PART.** MetaQA: W0 completes more than W1 at every rung (+21.3, +13.3, +14.9, ABOVE at all three). WebQSP: W0 more at every rung by the point estimate (+12.0 LEVEL, +17.8 ABOVE, +11.7 LEVEL). **HotpotQA: W0 more at 100/50 (+3.2) and 300/150 (+6.7) but W1 more at 1000/500 (-4.2, 24 questions, LEVEL): the expectation fails there by its point estimate.** MuSiQue shows no order (-6.4, +1.6, +7.7, all LEVEL). 2Wiki's half (W1 at least W0 at every rung) is pending.
- **XC4 FAILED AT 300/150 and HELD AT 1000/500.** G-P holds at 100/50 on all 4 (100 percent). **At 300/150 two datasets fall below 99.5 percent pooled (HotpotQA 98.4, WebQSP 98.7)**, which I expected to hold; the unproven programs there are HotpotQA W1 (1 of 31) and WebQSP W1 (2 of 75). At 1000/500 **all 4 read are below** (MuSiQue 91.3, MetaQA 99.3, HotpotQA 85.5, WebQSP 90.0), where I expected at least one; the recall at that rung is read on every dataset, the completion on one cell.
- XC5 to XC7 are Part B9's and are not scored here.

**What this means.** The cap, not the budget and not the program, is where R8's flat cosine loses reach: widening it lifts the candidate recall by 20 to 64 points on the 4 datasets read. **But it does not lift the completion: at the same budget the completion at 300/150 and (on the proven questions) at 1000/500 is at or below the completion at 100/50 on every dataset read, so the flat ceiling stays R8's own 100/50 number.** The same program, given the same budget over more candidates, completes the golds of fewer questions, not more. The hop-1 need (e) says why a bigger cap alone would not have fixed it either: on WebQSP the 90th percentile of the need is the rank 7,337 for W0 (0.887 of the field at the ratio's own 90th percentile), and only 60 percent of the questions have their missed hop-1 golds inside the first 1,000. **What it leaves for Part B9:** whether a tree chooses the right branches where the flat cosine, given the same nodes, does not; the flat arm F of 17.5 is exactly that comparison at equal node counts. **Nothing changes in pools, links, PL or any model.** The 2Wiki row and the three tree rows that are still running are read after this commit.

### 17.13 The 2Wiki cap row filed (27 November 2026, 02:51 IST, 21:21 UTC): Part A9 on all five datasets; the two named readings do not change, **and 17.11's "what this means" is corrected for 2Wiki at 300/150**

**Row.** `r9cap-2wiki` finished (rc 0, 11 h 15 min on the host, 40,508 s in the process, 4 workers, peak 21.56 GB against the ask of 23.5), the record is `outputs/sem2/r9/2wiki/cap.json` (+ `.npz`, not committed), code unchanged (zsem2r9.py sha as in 17.11; the record carries it). **Gates, all PASS: G-E, G-R, G-Q, G-C, G-X, G-B, G-P.** G-X: at 100/50 the candidate recall equals R8's on every slice question for W0 and for W1 (1.0, 1.0). The reports `report_cap.json|md` and `report_cap_beside.json|md` are rerun on all five datasets (`report cap`, then `zsem2r9a.py`); nothing else in the code was touched. The slice is 75 of R8's 600 population (every eighth question).

**(a) The ladder on 2Wiki:**

| scorer | candidate recall % at 100/50, 300/150, 1000/500 | completion % at the same rungs (— = not read) | proven share % at the same rungs | unproven programs at the same rungs |
|---|---|---|---|---|
| W0 | 12.0 / 45.3 / 98.7 | 12.0 / — / — | 100.0 / 98.7 / 72.0 | 0 / 1 / 21 |
| W1 | 29.3 / 70.7 / 98.7 | 29.3 / — / — | 100.0 / 97.3 / 54.7 | 0 / 2 / 34 |

**(b) Beside, decided nothing: the completion against the rung 100/50 on the questions both programs were proven on** (the unproven programs are the large ones, so this is a selected subset, counted):

| scorer | rung against 100/50 | questions proven at both | completion at 100/50 on them % | completion at the rung on them % | difference, points [interval] | call | recall gain, all questions, points |
|---|---|---:|---:|---:|---|---|---:|
| W0 | 300/150 | 74 of 75 | 10.81 | 27.03 | 16.22 [6.76, 25.68] | ABOVE | 33.33 |
| W0 | 1000/500 | 54 of 75 | 11.11 | 3.7 | -7.41 [-14.81, -1.85] | BELOW | 86.67 |
| W1 | 300/150 | 73 of 75 | 30.14 | 43.84 | 13.7 [4.11, 23.29] | ABOVE | 41.33 |
| W1 | 1000/500 | 41 of 75 | 21.95 | 7.32 | -14.63 [-26.83, -2.44] | BELOW | 69.33 |

**(c) Proven shares per rung** (the pooled reading is the declared one; zsem2r9.py reads per scorer and rung; **no named reading depends on a flip**):

| rung | unproven programs | proven share pooled over both scorers | W0 | W1 | cells the pooled reading would not read although the per-scorer reading did |
|---|---:|---:|---:|---:|---|
| 100/50 | 0 | 100.0 | 100.0 | 100.0 | — |
| 300/150 | 3 | 98.0 | 98.7 | 97.3 | — |
| 1000/500 | 55 | 63.3 | 72.0 | 54.7 | — |

**(d) The hop-1 need of the missed golds** (the stand-in for an uncapped cap):

| dataset | questions with a missed hop-1 gold | scorer | need at most 100 / 300 / 1,000 (share of those questions, %) | need median, 90th percentile | need over field size, median, 90th percentile |
|---|---:|---|---|---|---|
| 2Wiki | 73 | W0 | 12.3 / 46.6 / 100.0 | 350, 581 | 0.362, 0.612 |
| 2Wiki | 73 | W1 | 30.1 / 72.6 / 100.0 | 162, 556 | 0.190, 0.499 |

**(e) W0 minus W1 at each rung, on the questions proven in both** (beside; XC3):

| rung | questions proven in both | completion W0 minus W1, points [interval] | call | recall W0 minus W1, points |
|---|---:|---|---|---:|
| 100/50 | 75 | -17.33 [-26.67, -6.67] | BELOW | -17.33 |
| 300/150 | 72 | -16.67 [-29.17, -5.56] | BELOW | -25.33 |
| 1000/500 | 30 | -3.33 [-16.67, 6.67] | LEVEL | 0.0 |

**(f) The named readings, with 2Wiki read (decided here, both unchanged).** **CAP_BINDS: TRUE for W0 and for W1** (the candidate recall at 1000/500 over 100/50 is +86.7 for W0 and +69.3 for W1 on 2Wiki, the largest of the five; five of five datasets now clear the 10 points, three are needed). **COMPLETION_FOLLOWS: FALSE for W0 and for W1**, as decided in 17.11(c): at 1000/500 the 2Wiki programs are proven in only 72.0 percent (W0) and 54.7 percent (W1), so the rung's completion is not read on 2Wiki, and the only cell of the ten that is read at that rung is still MetaQA W1 (-9.3). The 2Wiki row could not have changed the reading in either direction, as 17.11 said. **The flat ceiling, the best completion read over the rungs and scorers**, on 2Wiki is R8's own 100/50 cell (W1 29.3; W0 12.0), as on the other four (MuSiQue 33.3, MetaQA 42.7, HotpotQA 45.2, WebQSP 36.0) **where the reading is the declared one (read cells only); see the correction below for what the proven subset shows at 300/150.**

**(g) Correction to 17.11 ("what this means"), for 2Wiki at 300/150.** 17.11 wrote that the completion at 300/150 is at or below 100/50 on every dataset read, and that the flat ceiling stays R8's own 100/50 number. **That sentence is true on the four datasets it was written for and it is not true on 2Wiki.** At 300/150 the 2Wiki programs are proven on 74 of 75 questions (W0) and 73 of 75 (W1), so the rung is **not read** by the declared rule (pooled share 98.0 percent, per scorer 98.7 and 97.3, the threshold is 99.5); but on the questions proven at both rungs the completion **rises**: W0 10.81 to 27.03 (+16.2 points, ABOVE), W1 30.14 to 43.84 (+13.7 points, ABOVE), against recall gains of 33.3 and 41.3. So **2Wiki is the one dataset where widening the cap to 300/150 lifts the completion, by about a half (W0) and a third (W1) of what it lifts the recall**, and its 1000/500 proven subset falls again (-7.4 W0, -14.6 W1, both BELOW; 54 and 41 of 75 questions, the small programs). R8 had found 2Wiki the dataset where the cap is the loss (completion 92 to 99 percent of the candidate recall at 100/50) and where the stored label helped most; the ladder agrees on the first, and it says the lift is available only in the window where the programs are still provable. **The corrected statement:** on MuSiQue, MetaQA, HotpotQA and WebQSP the completion at 300/150 is at or below 100/50; on 2Wiki, on the proven subset (a selected one), it is above, and that cell is not read by the declared rule. **Not a named reading and not a verdict**; it does not change CAP_BINDS, COMPLETION_FOLLOWS or the flat ceiling as defined, it changes the sentence that said "every dataset read". An amendment that re-solves the unproven 300/150 programs (three on 2Wiki, one on HotpotQA, two on WebQSP) at a longer limit would make the rung read on those datasets; I name it and **do not declare it**: Part B9 is the question R9 was declared for.

**(h) The expectations of 17.7 that Part A9 can score, now on five datasets (a failed one is reported as failed).**
- **XC1 HELD on five of five.** W0's recall rises with the cap on every dataset (2Wiki 12.0, 45.3, 98.7) and the gain at 1000/500 is at least 15 points on all five; WebQSP's is 62.7, inside 50 to 75.
- **XC2 STILL CANNOT BE SCORED AS DECLARED** (1000/500's completion is read in one cell of ten); the direction I expected is there beside on all five datasets at 1000/500 (W0 -1.6, -17.6, -32.0, -9.9 and 2Wiki -7.4, against recall gains of +63.5, +42.7, +54.8, +20.0 and +86.7), and at 300/150 on 2Wiki the point estimate of the completion gain is 16.2 (W0) against half of the recall gain 16.7: just under half, but a gain, and not at or below zero.
- **XC3 FAILED IN PART (as in 17.11), 2Wiki's half HELD:** W1 completes at least as much as W0 at every rung by the point estimate (-17.33, -16.67, -3.33 points for W0 minus W1 at 100/50, 300/150, 1000/500; BELOW at the first two, LEVEL at the third). HotpotQA's half still fails at 1000/500.
- **XC4 FAILED AT 300/150 on three datasets of five and HELD AT 1000/500.** G-P holds at 100/50 on all five. At 300/150 the pooled proven share is below 99.5 on HotpotQA (98.4), WebQSP (98.7) and 2Wiki (98.0), which I expected to hold. At 1000/500 **all five are below** (MuSiQue 91.3, MetaQA 99.3, HotpotQA 85.5, WebQSP 90.0, 2Wiki 63.3), where I expected at least one.
- XC5 to XC7 are Part B9's and are not scored here.

**What this means.** Part A9 is complete on the five datasets. The cap binds on all five (candidate recall +20 to +87 points at 1000/500 over 100/50); the completion does not follow it at the declared rungs (it is not read at 1000/500 on nine cells of ten, and where it is read at 300/150 it is at or below 100/50 on four datasets and unread on the fifth); the exception worth carrying into Part B9's reading is **2Wiki at 300/150, where the proven subset shows a completion gain of 16 points for W0 and 14 for W1.** The time the programs need grows faster than the cap: the 2Wiki programs at 1000/500 have 89,000 to 111,000 candidates by their medians against a budget of 1,500, and 55 of 150 are not proven at 120 s, so a read at that rung needs a longer limit or a smaller program, not a different scorer. **Nothing changes in pools, links, PL or any model.**

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
