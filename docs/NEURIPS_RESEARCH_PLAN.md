# NeurIPS research blueprint: When is message passing worth it for retrieval?

> **Terminology and originality update (2026-08-29):** all new publication prose
> uses **QLS-MLP (Query-Local Structure MLP)**. The implementation key `sa_mlp`
> and all sealed SA-named files, tags, configurations, and hashes remain
> immutable. The rename avoids collision with the published TMLR method
> *SA-MLP*. The contemporaneous RTA paper also makes a broad
> message-passing-as-retrieval claim, so our narrow question is now: **for
> graph-aware candidate retrieval, how much apparent GNN benefit comes from the
> upstream retrieval prior, explicit query-local structural statistics, and
> learned propagation, respectively?** See
> `docs/TERMINOLOGY_AND_POSITIONING.md`.

> **Package A resumption update (2026-08-30):** A1, A2, and A3 are complete under
> new frozen protocols, without changing the sealed QLS/GNN models. A1 finds
> that validation-selected RRF improves over the stronger Dense/SPLADE source
> by 0.00–2.64 R@5 points. A2 finds genuine fixed-structure signal on WebQSP
> (+5.20 structural-summary versus RRF), MetaQA (+4.41), and HotpotQA (+1.01
> for RRF+PPR), but the best training-free rule still trails QLS-MLP by
> 3.89–17.97 points on four datasets. Rank fusion and one fixed rule therefore
> do not explain the full QLS result. See `docs/P0_RANK_CONTROLS_RESULTS.md`
> and `docs/P0_FIXED_STRUCTURAL_CONTROLS_RESULTS.md`. The subsequently frozen
> 19-parameter A3 control recovers part of the remaining gap on WebQSP,
> MetaQA, and HotpotQA, but still trails QLS by 2.55–11.77 R@5 points on four
> datasets. See `docs/P0_LINEAR_RANK_STRUCTURE_RESULTS.md`.

> **Project status (2026-09-01): Packages B, C, and E1 are running.** The
> workspace spend limit recorded on 2026-08-31 has lifted, and the three
> packages were resumed with every volume checkpoint preserved. The sealed
> fairness confirmation remains closed, and A1/A2/A3 live in a separate P0
> resumption lineage. Package A is closed to further test-driven tuning.
> Sections describing perturbations, prediction, theory, or launch commands
> remain plans until separately frozen. The frozen confirmation result and its
> decisions continue to take precedence.
>
> **A:** COMPLETE. **Candidate-generation headroom diagnostic:** COMPLETE.
> **B / C / E1:** RUNNING. **D:** GATED on all six Package C budget-400 cells.
> **E2:** GATED on a complete E1 screen, a validation-only rate selection, a
> commit, and a new freeze tag. **F:** sealed and unopened.
>
> **Headroom changes how results are read, not what they compute.** Every
> reported metric now has a companion ceiling `min(p, K) / g`. Candidate
> coverage `p / g` is not an oracle Recall@K and must never be quoted as one.
> MetaQA attains roughly 92% of its candidate ceiling and so is not primarily a
> ranking-model failure; WebQSP attains roughly 74% and retains the most
> genuine reranking headroom. See
> `docs/CANDIDATE_HEADROOM_RESULTS.md`.

> **Fairness-confirmation result and stop decision (2026-08-26):** the protocol
> frozen at `sa-mlp-confirmation-protocol-v1` is complete on all six datasets
> and five paired seeds. The unchanged QLS-MLP, seed-only MLP, and seed-aware
> selected GNN use identical frozen candidates, labels, losses, and seeds.
> Fixed graph summaries add R@5 signal beyond the seed prior on all three
> original GNN-win datasets: +6.86 points on MetaQA, +4.11 on WebQSP, and +3.70
> on HotpotQA, with positive seed/query intervals and Holm-adjusted significance.
> The seed prior explains at least 80% of the QLS gain only on HotpotQA (1/3), so
> the registered seed-prior explanation is rejected. Requiring both paired-seed
> and paired-query intervals to clear the -1 point margin, QLS-MLP is non-inferior
> to the seed-aware GNN on MetaQA and HotpotQA (2/3; registered substitution gate
> passed); WebQSP remains query-level inconclusive. QLS-MLP is 2.49--7.08x faster
> in warm-cache candidate reranking and saves 90--2,418 MiB of incremental peak
> GPU allocation across the six datasets, with its 9.3--20.5 second preprocessing
> and 0.030--2.835 GiB disk
> caches disclosed separately. The gate is now closed: no tuning of these models
> against test data is allowed. See `docs/SA_MLP_CONFIRMATION_RESULTS.md`.
>
> The supported claim is not “MLPs beat GNNs.” It is: **graph information is
> useful for retrieval, but in identifiable regimes fixed query-conditioned
> structural computation can substitute for learned message passing at much
> lower cached-reranking cost.** Any next mechanism, perturbation, or
> practical-width
> experiment must begin under a separate preregistered protocol.

> **Historical query-local structure MLP screen (superseded by the fairness result
> above):** the preregistered one-seed
> gate passed on MetaQA, WebQSP, and HotpotQA. QLS-MLP exceeds the frozen selected
> GNN by +4.49, +1.55, and +15.18 R@5 points, respectively. Query-local fixed
> descriptors drive the result; static global descriptors alone fail. This is
> screening evidence only. Because distance-0 revealed retrieval-seed membership
> to QLS-MLP but not to the frozen GNN, the subsequent confirmation preserved the
> QLS model unchanged and included both a seed-only non-message-passing control
> and a seed-aware GNN. The screen alone supported neither a graph-path mechanism
> nor a general QLS-MLP win. See
> `docs/SA_MLP_SCREEN_RESULTS.md`.

> **Historical confirmation freeze (now completed):** the follow-up in
> `docs/SA_MLP_CONFIRMATION_PROTOCOL.md` fixes all six datasets, five seeds,
> the unchanged QLS-MLP, a seed-only interaction control, and a seed-aware copy
> of each dataset's already-selected GNN family. R@5, paired seed/query
> intervals, Holm correction, a one-point non-inferiority margin, and systems
> accounting are fixed before new test access. Practical width selection and
> perturbations remain separate and prohibited during this confirmation.

> **Six-dataset stop-gate update (2026-08-26):** the contract frozen at
> `six-dataset-protocol-v1` is complete. Across five paired seeds, plain MLP
> wins R@5 on 2Wiki (+2.08 points, 95% CI +1.22 to +2.94) and MuSiQue (+1.10,
> +0.57 to +1.63); the validation-selected GNN wins on WebQSP (-4.05),
> HotpotQA (-4.64), and MetaQA (-7.71); SQuAD is neutral (-0.01, -0.27 to
> +0.25). MLP inference is 3.64--9.92x faster and uses 104--2,431 MiB less
> incremental peak GPU memory with essentially matched parameters. MetaQA's
> GNN advantage is largest at one hop and shrinks with hop count, so hop count
> is not the mechanism. The effectiveness, MetaQA-hop, and systems tables are
> in `docs/SIX_DATASET_RESULTS.md`. The preregistered stop condition is met; no
> mechanism or perturbation follow-up has launched.

> **Historical execution checkpoint (2026-08-25):** the exact protocol baseline is frozen
> at `paper-protocol-v0`. The controlling immediate experiment is now the
> complete-data relational-operator screen in
> `docs/OFFSET_OPERATOR_PROTOCOL.md`: plain MLP, Offset-MLP, K=4 Offset-MLP,
> GCN, GraphSAGE, GAT, and GIN on WebQSP, 2Wiki, and MuSiQue. Earlier pilot and
> rewiring results remain non-reportable. Large perturbation sweeps and the LODO
> predictor are deferred until this screen's result gate is reviewed. The
> completed screen is documented in `docs/OFFSET_SCREEN_RESULTS.md`: Offset has
> a strong efficiency advantage and a 2Wiki top-rank advantage, but not a
> cross-dataset R@5/R@20 advantage. No follow-up has launched.

> **Confirmation update:** the frozen five-seed gate in commit `2c869a8`
> replicates plain MLP's R@5 advantage over the preselected GNN on 2Wiki
> (+2.077 points, paired 95% CI +1.218 to +2.936) and MuSiQue (+1.103,
> +0.572 to +1.633). Width-16/32 models fail validation selection, so no
> 2-4x parameter claim is supported. Offset's R@20 deficit worsens
> monotonically with gold-set size, passing the gate for one preregistered
> coverage-aware K-direction objective. See `docs/CONFIRMATION_RESULTS.md`.

> **Coverage-variant update:** the single objective frozen in commit `364183f`
> failed on both datasets. Relative to original K=4, it changes R@20 by -2.357
> points on 2Wiki (paired 95% CI -2.794 to -1.919) and -2.332 on MuSiQue
> (-2.704 to -1.960), while also reducing R@5 and FullCov. No coefficient was
> tuned and no second variant was launched. See
> `docs/COVERAGE_VARIANT_RESULTS.md`.

## Executive decision

The paper must not claim that MLPs are universally better than GNNs. The sealed
fairness control establishes the narrower central result: graph-derived signal
is real, the retrieval prior alone does not explain it, and learned aggregation
is not always required to exploit it. The paper's conditional question is now:

> **Once retrieval has already happened, how much learned message passing is
> still necessary to rank graph-aware evidence?**

RTA asks whether message passing itself can be reframed and replaced by
retrieval. This project begins after retrieval: given a frozen upstream
retriever, candidates, and graph, it asks what learned neighborhood aggregation
adds beyond the retrieval prior and explicit query-local structure.

This is already supported empirically on the preregistered substitution gate,
not merely proposed. The remaining NeurIPS burden is mechanism and boundary:
explain why substitution succeeds on MetaQA/HotpotQA, is uncertain on WebQSP,
and misses the one-point margin on 2Wiki, while preserving the sealed main
table. New evidence must be collected under a new protocol; it cannot alter,
tune, or filter the completed confirmation.

The observed boundary is already concrete. QLS-MLP trails the seed-aware GNN by
1.44 R@5 points on 2Wiki, with both paired intervals below zero, and fails the
one-point non-inferiority test. MuSiQue's mean deficit is 0.96 points, but both
intervals extend below the margin, so it is not certified non-inferior. WebQSP
has a +0.27 point mean but a wide query-level interval and remains inconclusive.
HotpotQA (-0.53), MetaQA (-0.02), and the graph-light SQuAD control (-0.10) clear
the margin using both registered intervals. These failures and uncertainty are
part of the result, not targets for post-hoc model tuning.

The primary empirical setting is **Level-2 candidate reranking**, not C-RAG L1
partition routing. This is the cleanest controlled comparison because all four
models receive the same candidate pool, frozen embeddings, labels, and
query-level state. In the plain-MLP versus GNN control, topology is the GNN's
additional information. In the headline QLS-MLP versus seed-aware GNN control,
both sides receive the same retrieval seeds and graph; only the form of graph
computation differs—fixed query-local summaries versus learned propagation.

The target should be **NeurIPS 2027**. The 2026 full-paper deadline was May 6,
2026 and has passed. The 2027 call is not yet published; use the current
[NeurIPS 2026 main-track rules](https://neurips.cc/Conferences/2026/MainTrackHandbook)
as a planning prior, not as a promised 2027 requirement. The 2026 format allows
nine content pages and requires anonymized code/data and a paper checklist.

There are two legitimate submission routes. A **main-track** paper needs the
phase diagram, a held-out crossover predictor, or useful theory. An
**Evaluations & Datasets** paper can instead make a fully documented MPR-Bench
and four-level confound-separation protocol central. The latter is not an easier
fallback: it requires release-quality provenance, evaluation design, audits,
and limitations. Choose the route before resuming large experiments.

## 1. Exact scientific question

For query (q), frozen candidate set (C_q), candidate features (X_q), retrieval
seed indicator (z_q), candidate graph (G_q=(C_q,E_q,T_q)), and relevance vector
(y_q), compare the complete ladder:

\[
s_i^{\mathrm{MLP}} = f_\theta(x_i, u_q)
\]

\[
s_i^{\mathrm{seed}} = f_\theta(x_i, u_q, z_i)
\]

\[
\phi_i = \Phi(i; G_q,z_q),
\qquad
s_i^{\mathrm{QLS}} = f_\theta(x_i,u_q,z_i,\phi_i)
\]

with

\[
H_q^{(\ell+1)} = \mathrm{MP}_\theta(H_q^{(\ell)}, E_q, T_q),
\qquad
s_i^{\mathrm{GNN}} = g_\theta(h_i^{(L)},u_q).
\]

(u_q) is the same query state for every model, and (\Phi) is a frozen,
label-free function producing distance/path/connectivity/PPR summaries.
Candidate generation, labels, losses, negatives, validation budgets, and random
seeds are paired. The ladder isolates retrieval prior, fixed graph computation,
and learned neighborhood aggregation rather than collapsing them.

Define the primary gap in a regime (r) as:

\[
\Delta_r = R@5_{\mathrm{GNN},r} - R@5_{\mathrm{QLS},r}.
\]

The paper predicts the sign and magnitude of \(\Delta_r\), and estimates the
boundary where it crosses a pre-registered practical margin (initially one
percentage point).

## 2. What can and cannot be “proved”

An empirical paper cannot prove “MLPs are better than GNNs” for every graph,
task, or architecture. It can support four narrower, defensible statements:

1. **Clean-regime substitution or superiority:** on named datasets and a fixed
   protocol, fixed query-local structural computation matches or outperforms
   matched learned message passing within a preregistered margin.
2. **Mechanism:** the loss is associated with measurable neighborhood noise,
   rank collapse, cosine concentration, hub amplification, or gradient effects.
3. **Crossover:** controlled interventions produce regimes where the sign of
   \(\Delta\) changes.
4. **Prediction:** a model fitted only on training datasets/regimes predicts the
   sign/magnitude of \(\Delta\) on held-out datasets and topology shifts.

A small theory section can prove a crossover condition under a simplified
generative model. It cannot replace the real-data evidence.

## 3. Why L2 is the primary test

CRAG's L2 work motivated a candidate-level comparison, but the final standalone
contract does not execute C-RAG L2, `full_fd2`, expert fusion, or the router. It
uses selected frozen CRAG data products only: raw query/node embeddings, Dense
and SPLADE candidate IDs, labels/splits, and graph adjacency. This gives a sharp
test:

- same query;
- same pool and candidate ceiling;
- same frozen node/query embeddings and retrieval candidates;
- same listwise multi-positive objective;
- same query state and availability mask;
- same metric vector;
- explicit accounting of which model receives retrieval seeds, fixed graph
  summaries, or adjacency in its learned forward pass.

This also avoids a common confound in GNN-vs-MLP studies: comparing different
tasks, features, pooling functions, or negative sets.

## 4. Hypotheses

The two central hypotheses organize every future intervention:

> **H1 — Structural compressibility:** QLS-MLP should approach or match learned
> message passing when graph information useful for ranking a candidate can be
> compressed into seed membership, seed distance, path multiplicity or
> connectivity, and seed-conditioned diffusion.

> **H2 — Rich-content requirement:** GNNs should gain an advantage when ranking
> depends on richer neighbor content, interactions, typed/compositional
> relations, or higher-order information not adequately represented by fixed
> query-local summaries.

All directional phase predictions must be frozen before canonical runs:

| ID | Phase prediction | Expected sign of \(\Delta=GNN-QLS\) |
|---|---|---:|
| P1 | strong semantic features plus low positive-neighbor lift | negative |
| P2 | random/kNN edge additions or degree-preserving corruption | increasingly negative |
| P3 | high-degree or high-Gini hubs | negative; hub amplification |
| P4 | sparse typed edges with high relevance-path coverage | positive |
| P5 | feature masking or low feature SNR with useful topology | increasingly positive |
| P6 | deeper propagation under noisy topology | increasingly negative |
| P7 | a development-only utility predictor transfers to held-out regimes | predictable sign |

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

Develop a message-passing utility predictor—working name **Retrieval
Neighborhood Utility (RNU)**—that estimates \(\Delta=GNN-QLS\) or the
probability that a GNN beats QLS by the practical margin. It must be trained and
tuned on development settings and evaluated on held-out regimes or datasets.
Do not lock the predictor architecture until the perturbation design and
available inference-time variables are audited.

Candidate inputs:

- Dense/SPLADE disagreement and candidate count;
- seed-to-candidate distance, connected-seed count, and path redundancy;
- PPR concentration/entropy;
- edge feature-alignment lift over random pairs;
- degree mean, tail, Gini, and max/mean hubness;
- density and reciprocity;
- native/kNN edge proportion and edge-type distribution;
- candidate-pool size and graph connectedness;
- semantic-neighborhood coherence;
- disagreement between graph neighbors and frozen semantic ranking.

Candidate ceiling, seed recall, positive-neighbor lift, and relevant-neighbor
coverage may be reported as label-dependent explanatory diagnostics, but they
cannot be deployment-time predictor inputs.

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

### Canonical execution packages

The detailed execution scope is consolidated in
`docs/PAPER_READINESS_AND_REAL_WORLD_FUTURE_WORK.md`:

```text
P0  A semantic/structural controls
P0  B edge provenance (mandatory)
P0  C candidate/context budgets 50/100/200/400
P0  D uncached post-retrieval systems evaluation

P1  E robustness perturbations -> phase diagram -> utility predictor

P2  freeze A-E hypotheses/protocol
P2  F untouched external confirmation
```

External confirmation is last by design. Its test outcomes must remain unseen
until the controls, graph rule, phase hypotheses, predictor inputs, and systems
protocol are frozen.

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

The standalone contract is complete for 2Wiki, MuSiQue, HotpotQA, MetaQA,
WebQSP, and SQuAD with stable query IDs, frozen splits, candidates, labels,
embeddings, and flattened adjacency. The next export must preserve/recover raw
edge source and direction without changing the sealed union graph or the
read-only CRAG source.

Role of each dataset:

- **2Wiki and MuSiQue:** multi-positive/multi-hop text retrieval; primary clean
  historical plain-MLP-win regimes and current fixed-versus-learned boundary.
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

The one-test-call rule applies within each sealed protocol. Because the project
used the same benchmark families across sequential screens and confirmations,
the final paper must include one new external or hidden holdout whose outcomes
were not observed during method design. Existing six-dataset results remain
development/replication evidence rather than the sole fresh confirmation.

## 8. Models and controls

### Primary pair

- QLS-MLP with fixed query-local structural summaries;
- the seed-aware GNN family selected by the frozen validation-only rule;
- trainable-parameter gap within the registered matched range;
- plain MLP and seed-only MLP retained in every headline table as decomposition
  controls.

### Architecture robustness

- GraphSAGE;
- GATv2;
- GIN;
- R-GCN or an equally simple typed-edge model when canonical edge types exist.

Do not run every architecture at every phase-diagram point. Establish the full
diagram with the primary QLS-MLP/seed-aware-GNN pair, then validate
representative fixed-structure-win, crossover, and GNN-win anchors with the
other architectures.

### Diagnostic controls

- Dense rank, SPLADE rank, and locked equal RRF: tests whether simple upstream
  rank fusion explains the learned gain.
- Seed distance/PPR alone, RRF+PPR, and a linear QLS head: tests whether either a
  parameter-free rule or linear scorer already captures the structural gain.
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
8. Rebuild candidate-induced topology and QLS features on demand and require
   numerical parity with the packed caches before uncached timing.
9. Verify that edge-source sidecars reproduce the frozen union adjacency before
   native-only/kNN-only ablation.
10. Verify that the fresh external setting contains all four required objects:
    queries, frozen candidate rankings, relevance labels, and either native
    topology or a label-free graph-construction rule frozen before test access.

Failure of any gate blocks all paper runs.

### E1. Clean L2 comparison

For each complete Tier-A dataset, retain the full four-level ladder: plain MLP,
seed-only MLP, QLS-MLP, and the seed-aware validation-selected GNN. Additional
GCN/SAGE/GATv2/GIN families are robustness controls, not test-selected headline
models. Report R@1/5/20, MRR, nDCG@10, FullCov@20, candidate ceiling,
conditional recall, time/query, train time, peak VRAM/RAM, parameters, and FLOPs.

Run the same primary pair at preregistered candidate budgets 50, 100, 200, and
400 (or the full union when fewer than 400 unique candidates exist). Use the
same budget for every model and report candidate ceiling, R@5, R@20, induced
node/edge count, QLS computation, GNN computation, and total latency together.

Edge provenance is a mandatory causal control, not optional metadata. Re-export
native/title/KB structural edges and embedding-derived kNN edges into separate
sidecars, prove that their union reconstructs the frozen adjacency, and run
native-only, kNN-only, and union conditions for both QLS-MLP and the seed-aware
GNN. This tests whether graph gains merely recycle the same embedding geometry
already available to the ranker.

The primary endpoint is paired query-level R@5 for the seed-aware selected GNN
minus QLS-MLP. The plain and seed-only models retain their causal-decomposition
roles. Do not choose the “best GNN” on test. Architecture-specific hypotheses
are secondary.

### E2. One-axis phase sweeps

Use fixed features and labels while varying one axis:

| Intervention | Levels | Question isolated |
|---|---|---|
| removal of high-ranked seeds | preregistered top-seed fractions | dependence on seed quality |
| Dense/SPLADE disagreement | controlled overlap strata | retriever-prior conflict |
| irrelevant-candidate injection | fixed replacement fractions | robustness to candidate noise |
| random edge addition | 0, 10, 25, 50, 100% of \(|E|\) | unstructured neighbor noise |
| degree-preserving rewiring | 0, 10, 25, 50, 100% | semantics without degree change |
| edge deletion | 0, 10, 25, 50, 75% | graph sparsity/connectivity |
| edge-type shuffling | 0, 10, 25, 50, 100% | relational semantics |
| native-edge removal | by source/fraction | loss of genuinely relational topology |
| kNN density | preregistered neighbor counts | semantic topology strength |
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

A full Cartesian product is wasteful and statistically awkward. Freeze the
design size after feasibility profiling; a Sobol or Latin-hypercube design can
cover:

\[
\text{feature SNR}\times\text{seed quality}\times\text{retriever disagreement}
\times\text{candidate noise}\times\text{degree/hubness/density}
\times\text{native:kNN ratio}\times\text{train size}.
\]

Record candidate ceiling, seed recall, seed-to-candidate distance, connected
seed count, path redundancy, PPR concentration/entropy, clustering, degree/hub
exposure, semantic-neighborhood coherence, and graph/semantic agreement.
Label-dependent quantities such as candidate ceiling and seed recall are oracle
diagnostics, not deployable predictor inputs. Reserve held-out generated regimes
and repeat with an entire dataset held out. Do not specify the final predictor
architecture before preregistration.

### E4. OOD graph transfer

- train model hyperparameters on text graphs; test a held-out text graph;
- train RNU on all but one dataset; predict the held-out dataset;
- test feature encoder shift while keeping topology fixed;
- test topology shift while keeping features fixed;
- entity-disjoint MetaQA/WebQSP split;
- train on low-noise regimes, test higher noise and vice versa.

NQ, MS MARCO, or BEIR may provide fresh queries, but none is automatically an
external graph-retrieval confirmation. A valid external setting must also have
frozen candidates, labels, and graph structure. When native topology is absent,
freeze a label-free rule such as Wikipedia hyperlinks/title mentions, citation
links, or KB triples before viewing external test outcomes.

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

The completed QLS-MLP confirmation measures warm-cache candidate reranking, not
an uncached post-retrieval path for unseen query embeddings. A resumed systems
study must report two separate views: cached reranker latency and uncached
post-retrieval latency. The latter begins from an upstream query embedding plus
Dense/SPLADE ranked candidate IDs and charges RRF/union construction, candidate
gathering, induced-subgraph extraction, method-specific graph work, model
inference, and top-K selection. Query encoding and initial retrieval are shared
upstream services outside this paper's scope. Corpus-static node features may
remain offline, but no cache keyed by a query or its candidate set may be used
in the unseen-embedding condition.

The GNN must be charged for on-demand candidate-induced topology construction;
QLS-MLP must be charged for the same construction plus its query-local
distance/path/PPR summaries. Report batch-1 and batch-16 p50/p95/p99,
throughput, peak GPU/RSS, static storage, cold start, and update/invalidation
costs. The detailed contract is in
`docs/RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md`.

### E6. Dense/SPLADE fusion control

The frozen candidate arrays retain Dense and SPLADE ranks but not raw scores.
Use these ranks for a locked equal-RRF baseline with constant 60. Report Dense,
SPLADE, equal RRF, and any validation-selected weighted RRF on the identical
candidate union. Equal RRF changes ordering, not candidate ceiling. If RRF
scores or RRF-derived top-K seeds are supplied to learned models, supply them
identically to QLS-MLP and seed-aware GNN and retrain both under a new frozen
protocol. Do not splice those runs into the sealed confirmation.

## 10. Phase-diagram statistics

### Primary test

For each dataset/regime, compute a paired query vector of
\(R@5_{GNN}-R@5_{QLS}\). Report a 95% paired bootstrap confidence interval and
the mean over pre-registered seeds. Define:

- **fixed-structure win:** upper confidence bound below (-1\) percentage point;
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

No theorem has been established. The following is a possible future analytical
starting point, not a claim about the completed experiments.

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

If this abstraction proves useful, the target would be a transparent crossover
inequality whose variables map to measured utility statistics. A result that
merely restates “high homophily is good” would be insufficient because the
closest prior work already covers that idea.

## 13. Falsification criteria

The project should change direction if any of the following occurs:

- GNN gains disappear after correcting a gradient or candidate-pool bug.
- QLS/fixed-structure gains vanish under parameter/validation parity.
- no GNN wins even on typed, high-signal, feature-degraded positive controls.
- the crossover is architecture-specific and does not replicate with at least
  one of SAGE/GAT/GIN.
- RNU fails leave-one-dataset-out and only interpolates perturbations.
- mechanism metrics do not track performance better than obvious baselines.
- results depend on incomplete query caches or test-selected hyperparameters.

If clean fixed-structure wins are robust but RNU/OOD prediction fails, the work is better
positioned as a strong WWW/KDD retrieval analysis than a NeurIPS paper.

## 14. Acceptance gates

### Gate A — correctness

All E0 tests pass; standalone plain/seed/QLS contract parity is documented; no
dead GNN gradients.

### Gate B — real bidirectional phase diagram

At least two real datasets contain preregistered fixed-structure-win regimes and at least
one real/controlled regime contains a replicated GNN win. Crossovers survive
parameter matching and two GNN families.

### Gate C — prediction

RNU predicts held-out sign materially above majority-class and homophily-only
baselines, with useful calibration and crossover-location error.

### Gate D — mechanism

At least one mechanistic quantity explains failure beyond raw homophily and
degree and behaves consistently across datasets.

### Gate E — generality and scale

One fresh external non-CRAG retrieval graph, untouched during method design,
replicates the central relationship, and the paper reports honest
compute/memory boundaries.

### Gate F — novelty

The introduction and experiments explicitly distinguish this work from node
classification phase analyses, graph-aware MLPs such as
[Graph-MLP](https://arxiv.org/abs/2106.04051), propagation-at-test approaches,
the published SA-MLP, RTA, and GNN/MLP training accelerators. The novelty is the
retrieval-prior/fixed-structure/learned-propagation decomposition, candidate
ranking setting, edge-provenance controls, a predictive crossover, and
mechanistic/OOD validation—not merely an MLP baseline.

### Gate G — ranker-serving and topology provenance

The paper reports uncached post-retrieval latency from unseen query embeddings
and upstream ranked IDs, with query-specific topology/features built on demand.
It also separates native/title/KB edges from embedding-kNN edges and shows
native-only, kNN-only, and union anchors. Query encoding and initial retrieval
remain explicitly out of scope.

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

1. clean four-level plain/seed/QLS/GNN retrieval and efficiency;
2. RNU versus homophily-only/degree-only baselines;
3. OOD graph/encoder transfer;
4. control models and mechanism ablations.

Everything else belongs in the appendix: all seeds, hyperparameters, full
perturbation grids, per-dataset plots, gradient audits, and additional models.

## 16. Deferred execution roadmap

The schedule below is retained as a planning artifact. It is not active while
the project is paused. If the project resumes, unfinished items require a new
preregistered protocol and must not modify the sealed confirmation.

### Weeks 1–2: canonicalize L2

- completed: freeze `paper-protocol-v0` and execute the contract-only WebQSP,
  2Wiki, and MuSiQue Modal pilot;
- completed: canonical standalone train/validation/test manifests and candidate
  contracts for all six datasets;
- completed: MLP/GNN parity, compatibility, and graph-induction tests;
- outstanding: recover typed/native/kNN edge provenance into standalone
  sidecars without changing the frozen union adjacency.

### Weeks 3–5: clean study

- completed: primary six-dataset plain-MLP/GNN runs;
- completed: validation-only GNN family selection and five-seed four-level
  plain/seed-only/QLS/GNN confirmation;
- completed: frozen practical margin, paired statistics, and clean-result table;
- outstanding: no perturbation or new architecture run until the six
  submission-critical packages are separately preregistered.

### Weeks 6–9: Packages A–D

- complete equal/weighted RRF and structural/simple controls;
- recover and verify edge provenance, then run native-only/kNN-only/union;
- run the shared 50/100/200/400 context-budget study;
- complete cached and uncached post-retrieval systems accounting.

### Weeks 10–14: Package E phase diagram

- run one-axis seed/candidate/topology/feature interventions;
- execute Sobol regimes;
- run mechanistic probes and compute profiles;
- fit RNU and simple baselines;
- leave one regime family and one dataset out;
- derive/validate crossover estimates.

### Weeks 15–17: explanation and protocol freeze

- finish ranking-SNR crossover derivation;
- freeze H1/H2, predictor inputs, metrics, graph rule, and all Package F hashes;
- do not inspect the external outcome during this stage.

### Week 18: Package F untouched confirmation

- evaluate the fully locked method once on a fresh external holdout;
- report the predictor and primary hypotheses exactly as preregistered.

### Weeks 19–22: paper hardening

- rerun headline cells with 10 seeds if confidence intervals are close;
- freeze tables from machine-readable result manifests;
- independent leakage/reproducibility audit;
- anonymous one-command supplement under the size limit;
- draft rebuttal-risk matrix and run only pre-identified missing controls.

## 17. Archived launch examples — do not run while paused

The following commands are retained only for provenance. No run is currently
scheduled. The completed engineering pilot was reproducible through the
restricted launcher:

```bash
python experiments.py run pilot3 --backend modal --intervention clean --rate 0
```

The output must say `NOT_PAPER_VALID_PILOT`. Its only purposes were verifying
gradient flow, memory, runtime, feature parity, and metric serialization.
Degree-preserving rewiring and the LODO predictor are deferred future work, not
an immediate gate.

A future scientifically valid resumption would require canonical split export
and a new preregistration before any command such as the following is used:

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
substrate study rather than a fundamental graph-retrieval result. C-RAG itself
is not a contribution of this standalone paper.
