# NeurIPS research blueprint: When is message passing worth it for retrieval?

> **Execution checkpoint (2026-08-25):** the exact protocol baseline is frozen
> at `paper-protocol-v0`. The three-dataset Modal run and initial 25%
> degree-preserving-rewiring intervention are strictly non-reportable pilots;
> see `docs/PILOT3_RESULTS.md`. They validate the comparison machinery and show
> a shrinking GCN R@5 margin, including a near-neutral MuSiQue regime, but no
> robust MLP-winning crossover. The LODO predictor remains gated.

## Executive decision

The paper should not claim that MLPs are universally better than GNNs. That is
both scientifically unlikely and too easy to refute. The paper should establish
a conditional result:

> In query-conditioned retrieval, the value of message passing is predictable
> from neighborhood task signal, feature strength, topology corruption, edge
> semantics, degree concentration, and supervision. We identify and predict the
> crossover between regimes where message passing helps and regimes where it
> destroys useful candidate representations.

The primary empirical setting is **Level-2 candidate reranking**, not C-RAG L1
partition routing. This is the cleanest controlled comparison because both
models receive the same candidate pool, the same expert evidence, the same
labels, and the same query-level state. The GNN receives exactly one additional
input: edges among the candidates.

The target should be **NeurIPS 2027**. The 2026 full-paper deadline was May 6,
2026 and has passed. The 2027 call is not yet published; use the current
[NeurIPS 2026 main-track rules](https://neurips.cc/Conferences/2026/MainTrackHandbook)
as a planning prior, not as a promised 2027 requirement. The 2026 format allows
nine content pages and requires anonymized code/data and a paper checklist.

## 1. Exact scientific question

For query (q), frozen candidate set (C_q), candidate features (X_q),
candidate graph (G_q=(C_q,E_q,T_q)), and relevance vector (y_q), compare:

\[
s_i^{\mathrm{MLP}} = f_\theta(x_i, u_q)
\]

with

\[
H_q^{(\ell+1)} = \mathrm{MP}_\theta(H_q^{(\ell)}, E_q, T_q),
\qquad
s_i^{\mathrm{GNN}} = g_\theta(h_i^{(L)},u_q).
\]

(u_q) is the same query/expert state for both models. Candidate generation is
frozen. Training/evaluation labels, losses, negatives, validation budgets, and
random seeds are paired. Thus the causal intervention is the use of
neighborhood aggregation.

Define the primary gap in a regime (r) as:

\[
\Delta_r = R@5_{\mathrm{GNN},r} - R@5_{\mathrm{MLP},r}.
\]

The paper predicts the sign and magnitude of \(\Delta_r\), and estimates the
boundary where it crosses a pre-registered practical margin (initially one
percentage point).

## 2. What can and cannot be “proved”

An empirical paper cannot prove “MLPs are better than GNNs” for every graph,
task, or architecture. It can support four narrower, defensible statements:

1. **Clean-regime superiority:** on named datasets and a fixed protocol, the
   paired MLP has higher expected retrieval performance than matched GNNs.
2. **Mechanism:** the loss is associated with measurable neighborhood noise,
   rank collapse, cosine concentration, hub amplification, or gradient effects.
3. **Crossover:** controlled interventions produce regimes where the sign of
   \(\Delta\) changes.
4. **Prediction:** a model fitted only on training datasets/regimes predicts the
   sign/magnitude of \(\Delta\) on held-out datasets and topology shifts.

A small theory section can prove a crossover condition under a simplified
generative model. It cannot replace the real-data evidence.

## 3. Why L2 is the primary test

CRAG’s strongest L2 neural scorer is an MLP over candidate-level expert signals.
The current CRAG repository has GNNs for L1 partition encoding but no matched L2
candidate GNN. Creating the L2 pair gives a much sharper test:

- same query;
- same pool and candidate ceiling;
- same dense, lexical, relational, path, and adapter evidence;
- same listwise multi-positive objective;
- same query state and availability mask;
- same metric vector;
- topology is the only added information.

This also avoids a common confound in GNN-vs-MLP studies: comparing different
tasks, features, pooling functions, or negative sets.

## 4. Hypotheses

All directional hypotheses must be frozen before the canonical test runs.

| ID | Hypothesis | Expected sign of \(\Delta\) |
|---|---|---:|
| H1 | strong semantic/expert features plus low positive-neighbor lift | MLP wins |
| H2 | random or kNN edge additions at fixed features | increasingly MLP |
| H3 | degree-preserving topology corruption | increasingly MLP |
| H4 | high-degree or high-Gini hubs | MLP wins; hub nodes amplify |
| H5 | sparse, typed edges with high relevance-path coverage | GNN wins |
| H6 | node/expert feature masking or low feature SNR | GNN advantage grows |
| H7 | deeper message passing under noisy topology | MLP advantage grows |
| H8 | more training data helps GNN only when edge signal is positive | interaction |
| H9 | a train-only graph-utility predictor transfers across datasets | predictable sign |

Negative and positive controls are mandatory. If no deliberately favorable
regime makes a GNN win, the implementation or task construction is suspect.

## 5. Contribution package required for NeurIPS

### C1. Retrieval-specific phase diagram

Map \(\Delta\) against task-aligned graph properties rather than only global
class homophily. The closest conceptual prior,
[Luan et al., “When Do Graph Neural Networks Help with Node Classification?”](https://arxiv.org/abs/2304.14274),
studies node classification and node distinguishability. Our novelty must be
query-conditioned ranking with multi-positive relevance, strong pretrained
semantic features, typed retrieval edges, candidate ceilings, and OOD topology.

### C2. Predictive graph-utility index

Develop a simple, pre-training predictor—working name **Retrieval Neighborhood
Utility (RNU)**—that estimates \(\Delta\) or the probability that a GNN beats an
MLP by the practical margin. RNU must use train/validation graph statistics only
and be evaluated leave-one-dataset-out.

Candidate inputs:

- query-conditioned positive-neighbor lift;
- relevant-neighbor coverage;
- edge feature-alignment lift over random pairs;
- degree mean, tail, Gini, and max/mean hubness;
- density and reciprocity;
- edge-type distribution/purity;
- candidate-pool size and graph connectedness;
- node/expert feature SNR;
- training size;
- disagreement between graph neighbors and frozen semantic ranking.

Use a transparent linear interaction model and a monotonic GAM as primary
predictors. A large black-box meta-model would weaken the scientific story.

### C3. Mechanistic evidence

Measure layer by layer:

- entropy-based effective rank and stable rank;
- random-pair cosine mean/dispersion;
- normalized Dirichlet energy;
- representation norm versus degree;
- relevant-to-irrelevant neighbor mixing;
- gradient presence/norm for every convolution layer;
- attention or aggregation mass versus degree;
- query-positive score margin before and after each layer;
- performance stratified by local homophily and degree.

Effective rank is especially important because recent work argues that common
energy measures can miss harmful collapse; see
[“Are We Measuring Oversmoothing in Graph Neural Networks Correctly?”](https://arxiv.org/abs/2502.04591).

### C4. Strict paired benchmark and reproducible release

Release canonical candidate IDs, immutable split manifests, perturbation seeds,
query-level predictions, graph statistics, efficiency logs, and one command per
main table. The artifact must be independent of C-RAG.

## 6. Data tiers

### Tier P: engineering pilots (never paper evidence)

The current CRAG L2 signal caches are useful to validate code and compute cost,
but contain only existing test-query subsets. `export_crag_l2.py` marks them as
pilot-only. `run_l2_pair.py` refuses to train unless an explicit engineering
override is supplied, and override results are marked `NOT_PAPER_VALID_PILOT`.

Current counts observed in the read-only cache:

| Dataset | Cached queries | Signals | Paper status |
|---|---:|---:|---|
| 2Wiki-clean | 1,500 | 5 | pilot only |
| MuSiQue-clean | 1,995 | 4 | pilot only |
| HotpotQA-clean | 2,000 | 4 | incomplete; pilot only |
| MetaQA | 2,000 | 4 | pilot only |
| WebQSP | 159 | 5 | too small/incomplete; pilot only |

### Tier A: canonical retrieval datasets

Rebuild 2Wiki, MuSiQue, HotpotQA, MetaQA, WebQSP, and SQuAD with complete stable
query IDs and official splits where available. Freeze candidate generation
before any MLP/GNN comparison. Preserve raw edge type and direction.

Role of each dataset:

- **2Wiki and MuSiQue:** multi-positive/multi-hop text retrieval; primary clean
  MLP-win candidates.
- **HotpotQA:** scale and hubness stress test after complete reconstruction.
- **MetaQA:** sparse typed relational positive control where GNNs may win.
- **WebQSP:** OOD KB retrieval and typed-edge semantics; requires complete
  export and entity-disjoint evaluation.
- **SQuAD:** mostly single-positive negative control; graph structure should
  have low marginal value.

### Tier B: external validity

Use one non-CRAG retrieval graph for every major conclusion. Suitable choices
are a public citation/document retrieval task and a public KG link/entity
retrieval task. Node-classification benchmarks may appear only as an appendix
bridge to prior literature; making them central would dilute the retrieval
contribution.

## 7. Split and leakage contract

1. Use official query splits whenever defined.
2. When official train data must be partitioned, hash stable query IDs; never
   shuffle an in-memory ordering whose construction may change.
3. Create a second OOD split by entity/topic/time/graph component, not random
   query assignment.
4. Candidate generation and embedding models are frozen before reranker tuning.
5. Compute feature normalization on training candidates where the expert is
   present; missing values never enter the statistics.
6. Tune hyperparameters and stopping on validation only.
7. Evaluate each final seed on test once.
8. Never construct a test graph using test relevance labels. Label-guided graphs
   may be reported only as explicitly marked oracle diagnostics.
9. Record candidate ceiling separately. End-to-end recall divides by all golds;
   conditional recall divides by golds present in the pool.
10. Store query-level predictions so every comparison is paired.

## 8. Models and controls

### Primary pair

- residual candidate MLP;
- residual GCN with 1/2/3/4 layers;
- trainable-parameter gap within 5%.

### Architecture robustness

- GraphSAGE;
- GATv2;
- GIN;
- R-GCN or an equally simple typed-edge model when canonical edge types exist.

Do not run every architecture at every phase-diagram point. Establish the full
diagram with the primary GCN/MLP pair, then validate representative MLP-win,
crossover, and GNN-win anchors with the other architectures.

### Diagnostic controls

- MLP plus degree/local graph statistics: tests whether graph information helps
  without learned message propagation.
- GNN with self-loops only: architecture/training control equivalent to no
  cross-node messages.
- GNN with shuffled edges: tests topology specificity.
- GNN with frozen random message layers: tests parameter-count explanations.
- Simple non-parametric smoothing before MLP: separates smoothing from learned
  propagation.
- Gated residual \((1-\alpha)h+\alpha\,\mathrm{MP}(h)\): diagnostic estimate of
  optimal message strength, not the main claimed method.

### Training parity

- same input feature builder;
- same ListNet/multi-positive loss;
- same candidate pool and negatives;
- same optimizer family and learning-rate search budget;
- same epochs/patience and validation endpoint;
- same five primary seeds (ten for headline close calls);
- identical mixed precision and sampling settings;
- full gradient through positive and negative candidates;
- convolution gradient audit on the first step of every run;
- both parameter-matched and compute-matched analyses.

## 9. Experimental program

### E0. Correctness and parity gates

Run before scientific sweeps:

1. Overfit 20 synthetic queries with both models.
2. Verify every GNN convolution parameter receives a finite nonzero gradient.
3. With only self-loops, confirm GNN and peer MLP have comparable capacity.
4. Permute candidate ordering and confirm rankings/metrics are equivariant.
5. Verify candidate graphs never contain global nodes outside the pool.
6. Reproduce the frozen CRAG L2 MLP within tolerance using the neutral feature
   builder, or explain any intentional feature difference.
7. Confirm train-only normalization and a single final test call.

Failure of any gate blocks all paper runs.

### E1. Clean L2 comparison

For each complete Tier-A dataset, run MLP, GCN, SAGE, GATv2, GIN and typed GNN
where applicable. Report R@1/5/20, MRR, nDCG@10, FullCov@20, candidate ceiling,
conditional recall, time/query, train time, peak VRAM/RAM, parameters, and FLOPs.

The primary endpoint is paired query-level R@5 for GCN minus MLP. Do not choose
the “best GNN” on test. Architecture-specific hypotheses are secondary.

### E2. One-axis phase sweeps

Use fixed features and labels while varying one axis:

| Intervention | Levels | Question isolated |
|---|---|---|
| random edge addition | 0, 10, 25, 50, 100% of \(|E|\) | unstructured neighbor noise |
| degree-preserving rewiring | 0, 10, 25, 50, 100% | semantics without degree change |
| edge deletion | 0, 10, 25, 50, 75% | graph sparsity/connectivity |
| edge-type shuffling | 0, 10, 25, 50, 100% | relational semantics |
| semantic/kNN edge removal | by edge type | synthetic versus curated topology |
| feature masking | 0, 25, 50, 75, 100% | node/evidence strength |
| Gaussian feature noise | 30, 20, 10, 5, 0 dB | continuous feature SNR |
| train fraction | 5, 10, 25, 50, 100% | sample efficiency |
| message-passing depth | 0, 1, 2, 4, 8 | collapse with depth |
| hub injection | controlled max/mean and Gini | hub amplification |
| density/fanout | fixed degree targets | compute and oversquashing |

Every topology operation must record achieved—not merely requested—statistics.
Degree-preserving rewiring is critical: it can show that a loss comes from edge
meaning rather than degree distribution.

### E3. Joint phase diagram

A full Cartesian product is wasteful and statistically awkward. Use 128 Sobol
or Latin-hypercube regimes per dataset over:

\[
\text{feature SNR}\times\text{positive-neighbor lift}\times\text{degree}
\times\text{hubness}\times\text{density}\times\text{edge-type purity}
\times\text{train size}.
\]

Reserve 25% of generated regimes as an untouched design test set. Fit RNU on
the remaining regimes using training/validation statistics. Repeat with an
entire dataset held out.

### E4. OOD graph transfer

- train model hyperparameters on text graphs; test a held-out text graph;
- train RNU on all but one dataset; predict the held-out dataset;
- test feature encoder shift while keeping topology fixed;
- test topology shift while keeping features fixed;
- entity-disjoint MetaQA/WebQSP split;
- train on low-noise regimes, test higher noise and vice versa.

OOD model performance and OOD **predictor** performance are distinct results.

### E5. Scale and systems boundary

For graph sizes from controlled subsamples to full Hotpot/WebQSP, report:

- total training time;
- examples/second;
- peak VRAM and RAM;
- preprocessing/graph construction time;
- inference time/query;
- edge storage;
- failure/OOM boundary.

Use full-batch computation where feasible and a documented neighbor-sampling
path at scale. Do not compare full-batch MLP with sampled GNN without a second
matched-sampling control.

## 10. Phase-diagram statistics

### Primary test

For each dataset/regime, compute a paired query vector of
\(R@5_{GNN}-R@5_{MLP}\). Report a 95% paired bootstrap confidence interval and
the mean over pre-registered seeds. Define:

- **MLP win:** upper confidence bound below (-1\) percentage point;
- **GNN win:** lower confidence bound above (+1\) percentage point;
- **practical tie:** confidence interval lies inside the equivalence band;
- **uncertain:** all other cases.

Use two one-sided tests for equivalence, not “non-significant = equal.”

### Across-regime model

Fit a hierarchical model with dataset and seed random effects and fixed effects
for graph statistics/interactions. Report coefficients, uncertainty, partial
effects, and leave-one-dataset-out prediction.

### Multiple comparisons

- one pre-registered primary metric and primary model pair;
- Holm correction for the small family of secondary confirmatory hypotheses;
- label broad exploratory grids as exploratory;
- publish all regimes, not only crossovers.

### Predictor endpoints

- regression MAE/RMSE for \(\Delta\);
- sign accuracy and AUROC for practical GNN win;
- Brier score/calibration;
- crossover-location error along held-out corruption trajectories;
- leave-one-dataset-out confidence intervals.

## 11. Mechanism and mediation analyses

For each layer and regime, capture representations before propagation, after
aggregation, and after the residual update. Plot the performance gap alongside:

1. effective-rank fraction;
2. cosine concentration;
3. positive-negative score margin;
4. relevant-neighbor signal;
5. degree-norm/attention correlation;
6. edge-type message mass;
7. gradient norm by layer.

Then test whether changes in effective rank, score margin, or hub amplification
mediate the relationship between topology corruption and \(\Delta\). Treat
mediation as explanatory evidence, not causal proof, because representations
are post-treatment variables.

Key stratified plots:

- \(\Delta\) by candidate degree decile;
- \(\Delta\) by query-conditioned homophily decile;
- \(\Delta\) by answer-set size/hop count;
- \(\Delta\) by feature confidence;
- \(\Delta\) for hub versus non-hub positives;
- layerwise rank/cosine trajectory for clean, crossover, and corrupted graphs.

## 12. Theory work package

Start with a binary relevance model. Let node feature
(x_i=y_i\mu+\epsilon_i\), where \(y_i\in\{-1,+1\}\) and
\(\epsilon_i\sim\mathcal N(0,\sigma^2 I)\). Let a neighbor share the target
label with probability \(h\), so its expected label correlation is
\(\rho=2h-1\). For mean degree (d), consider a residual one-step message:

\[
z_i=(1-\alpha)x_i+\alpha\frac{1}{d}\sum_{j\in N(i)}x_j.
\]

The signal coefficient along \(\mu\) is
((1-\alpha)+\alpha\rho\), while independent feature noise contracts roughly
with (d) but label-mixture variance grows as \(\rho\) weakens. Derive the
condition under which the pairwise positive-negative ranking probability after
aggregation exceeds that of the unpropagated feature. Extend it with:

- edge corruption replacing \(\rho\) with effective \(\rho'\);
- nonuniform degree/hubs;
- multiple edge types with type-specific \(\rho_t\);
- finite feature SNR;
- query-conditioned relevance rather than global class labels.

The theorem target is a transparent crossover inequality whose variables map
to measured RNU statistics. A theorem that merely restates “high homophily is
good” is insufficient because the closest prior work already covers that idea.

## 13. Falsification criteria

The project should change direction if any of the following occurs:

- GNN gains disappear after correcting a gradient or candidate-pool bug.
- MLP gains vanish under parameter/validation parity.
- no GNN wins even on typed, high-signal, feature-degraded positive controls.
- the crossover is architecture-specific and does not replicate with at least
  one of SAGE/GAT/GIN.
- RNU fails leave-one-dataset-out and only interpolates perturbations.
- mechanism metrics do not track performance better than obvious baselines.
- results depend on incomplete query caches or test-selected hyperparameters.

If clean MLP wins are robust but RNU/OOD prediction fails, the work is better
positioned as a strong WWW/KDD retrieval analysis than a NeurIPS paper.

## 14. Acceptance gates

### Gate A — correctness

All E0 tests pass; CRAG MLP parity is documented; no dead GNN gradients.

### Gate B — real bidirectional phase diagram

At least two real datasets contain pre-registered MLP-win regimes and at least
one real/controlled regime contains a replicated GNN win. Crossovers survive
parameter matching and two GNN families.

### Gate C — prediction

RNU predicts held-out sign materially above majority-class and homophily-only
baselines, with useful calibration and crossover-location error.

### Gate D — mechanism

At least one mechanistic quantity explains failure beyond raw homophily and
degree and behaves consistently across datasets.

### Gate E — generality and scale

One external non-CRAG retrieval graph replicates the central relationship, and
the paper reports honest compute/memory boundaries.

### Gate F — novelty

The introduction and experiments explicitly distinguish this work from node
classification phase analyses, graph-aware MLPs such as
[Graph-MLP](https://arxiv.org/abs/2106.04051), propagation-at-test approaches,
and GNN/MLP training accelerators. The novelty is query-conditioned retrieval,
typed/noisy retrieval graphs, a predictive crossover, and mechanistic/OOD
validation—not merely an MLP baseline.

## 15. Paper structure and figure plan

Assuming the current nine-page NeurIPS main-text precedent:

| Section | Pages | Essential content |
|---|---:|---|
| Introduction | 1.0 | problem, gap, contributions |
| Setup and theory | 1.5 | paired retrieval task, crossover condition |
| RNU and protocol | 1.0 | statistics, predictor, fairness |
| Experimental setup | 1.0 | datasets, models, perturbations |
| Results | 3.5 | clean, phase diagram, prediction, mechanism, OOD |
| Related work | 0.6 | nearest distinctions |
| Limitations/conclusion | 0.4 | scope and implications |

Main figures:

1. 2-D phase diagram: feature SNR × positive-neighbor lift, colored by \(\Delta\).
2. Topology corruption trajectories with crossover confidence bands.
3. Layerwise effective rank/cosine/hub amplification for three anchor regimes.
4. Leave-one-dataset-out predicted versus observed \(\Delta\), calibrated by
   GNN-win probability.

Main tables:

1. clean matched MLP/GNN retrieval and efficiency;
2. RNU versus homophily-only/degree-only baselines;
3. OOD graph/encoder transfer;
4. control models and mechanism ablations.

Everything else belongs in the appendix: all seeds, hyperparameters, full
perturbation grids, per-dataset plots, gradient audits, and additional models.

## 16. Execution roadmap

### Weeks 1–2: canonicalize L2

- completed: freeze `paper-protocol-v0` and execute the contract-only WebQSP,
  2Wiki, and MuSiQue Modal pilot;
- regenerate stable L2 train/validation/test caches with candidate IDs;
- preserve typed/raw/synthetic edge provenance;
- implement MLP parity and graph induction tests;
- rerun the three clean datasets with five seeds only after their canonical
  manifests are frozen.

### Weeks 3–5: clean study

- complete primary six-dataset GCN/MLP runs;
- validate SAGE/GAT/GIN at clean anchors;
- lock primary hyperparameters and practical margin;
- write the clean-result table before perturbation exploration.

### Weeks 6–9: controlled phase diagram

- one-axis topology and feature sweeps;
- choose anchor regimes;
- run mechanistic probes and compute profiles;
- fix the joint-design ranges using training/validation data only.

### Weeks 10–13: joint design and predictor

- execute Sobol regimes;
- fit RNU and simple baselines;
- leave one regime family and one dataset out;
- derive/validate crossover estimates.

### Weeks 14–17: theory and external dataset

- finish ranking-SNR crossover derivation;
- replicate on one non-CRAG retrieval graph;
- complete typed-edge and entity-disjoint OOD experiments.

### Weeks 18–22: paper hardening

- rerun headline cells with 10 seeds if confidence intervals are close;
- freeze tables from machine-readable result manifests;
- independent leakage/reproducibility audit;
- anonymous one-command supplement under the size limit;
- draft rebuttal-risk matrix and run only pre-identified missing controls.

## 17. Immediate runnable sequence

The completed engineering pilot is reproducible through the restricted launcher:

```bash
python experiments.py run pilot3 --backend modal --intervention clean --rate 0
```

The output must say `NOT_PAPER_VALID_PILOT`. Its only purposes are verifying
gradient flow, memory, runtime, feature parity, and metric serialization. The
next pilot gate is degree-preserving rewiring at 0.10, 0.50, and 1.00; do not
start the LODO predictor from these results.

The first scientifically valid run occurs only after canonical split export:

```bash
python scripts/run_l2_pair.py \
  --data data/processed/2wiki_l2_canonical.pt \
  --output outputs/2wiki_l2_clean.json \
  --gnn gcn \
  --seeds 0 1 2 3 4
```

## 18. Bottom line

The NeurIPS-worthy result is not a table where an MLP wins. It is a reliable
answer to a model-selection question:

> Given candidate features and a retrieval graph, can we determine—before
> spending GNN compute—whether neighborhood propagation will add signal or
> amplify noise?

The paper becomes compelling if it supplies a matched benchmark, a bidirectional
phase diagram, a transferable predictor, a theory-aligned crossover, and
mechanistic evidence. Without those pieces, it remains an interesting CRAG
ablation rather than a fundamental graph-learning paper.
