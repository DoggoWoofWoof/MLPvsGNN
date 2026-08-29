# Graph-Aware Retrieval Without Learned Message Passing

> **Status: Research paused at the fairness-confirmed six-dataset checkpoint.**
>
> All six datasets, five paired seeds, the seed-aware fairness control, and the
> frozen statistical analysis are complete. The results are frozen. No further
> experiments are scheduled, and no model, feature, protocol, or test result
> should be changed while the project is paused.

This is a standalone research repository about the value and cost of graph
message passing for retrieval. It began with the question:

> **Do MLPs outperform GNN message passing for retrieval?**

The evidence led to a more precise and more useful question:

> **Does graph-aware retrieval actually require learned message passing, or can
> useful graph structure be exposed through fixed query-local structural
> summaries and consumed by a lightweight MLP?**

The final Structure-Aware MLP, abbreviated **SA-MLP**, is graph-aware and
**non-message-passing**. It uses deterministic structural features computed from
the frozen graph and retrieval seeds, but its learned forward pass never
aggregates neighbor embeddings or receives adjacency. It is therefore **not
topology-free**. Its trainable parameter count is approximately matched to the
GNN comparators; fewer parameters are not a contribution of this work.

## Current Research Finding

> **Across six retrieval benchmarks, a non-message-passing MLP supplied with
> fixed query-local graph summaries recovers nearly all of the effectiveness of
> parameter-matched seed-aware GNNs, remaining within 1.44 Recall@5 points on
> every dataset while substantially reducing warm-cache candidate-reranking
> latency and GPU memory.
> Controlled seed-only ablations show that the structural gain is not explained
> solely by retriever-seed membership.**

The broader conceptual result is:

> **Graph structure can be useful for retrieval without requiring learned
> message passing.**

This does not mean that MLPs universally beat GNNs. GNNs retain small but
statistically supported advantages in some datasets and metrics; WebQSP is
numerically slightly positive for SA-MLP; and several regimes are effectively
at parity. The central finding is that learned message passing provides
surprisingly little incremental retrieval effectiveness once useful
query-local structural information is explicitly exposed.

The project does **not** claim that graphs are useless, SA-MLP is topology-free,
SA-MLP has fewer parameters, or message passing is never required.

## Frozen checkpoint

The authoritative checkpoint consists of:

- frozen protocol: tag `sa-mlp-confirmation-protocol-v1`, commit `00951ec`;
- legacy-candidate compatibility proof: tag
  `sa-mlp-confirmation-compat-v1`, commit `63b8f85`;
- frozen analysis compiler: tag `sa-mlp-confirmation-analysis-v1`, commit
  `7561049`;
- completed result and stopping decision: tag
  `sa-mlp-confirmation-results-v1`, commit `eca5cbd`;
- protocol document: [SA-MLP confirmation protocol](docs/SA_MLP_CONFIRMATION_PROTOCOL.md);
- complete five-seed tables: [SA-MLP confirmation results](docs/SA_MLP_CONFIRMATION_RESULTS.md);
- feature-safety evidence: [SA feature leakage audit](docs/SA_FEATURE_LEAKAGE_AUDIT.md);
- legacy candidate proof: [candidate compatibility audit](docs/LEGACY_CANDIDATE_COMPATIBILITY.md).

The earlier general comparison is separately frozen at
`six-dataset-protocol-v1` (`f012699`) and documented in
[the six-dataset results](docs/SIX_DATASET_RESULTS.md). Historical protocol tags
remain part of the record and must not be moved or deleted.

## Separation from C-RAG

This project is independent of the C-RAG systems paper. The original CRAG
repository remained read-only throughout extraction, experimentation, and this
documentation checkpoint.

This paper does not claim or reuse as its contribution:

- the C-RAG L1/L2/L3 architecture;
- the KL advisor/router;
- `full_fd2`;
- edge-semantic fusion;
- C-RAG traversal;
- unified C-RAG system claims.

Frozen embeddings, datasets, graphs, and candidate pools shared with CRAG are
experimental substrate only. Their provenance and the deliberately narrow
extraction boundary are recorded in
[the CRAG extraction audit](docs/CRAG_EXTRACTION_AUDIT.md). The standalone code,
protocols, models, controls, analyses, and claims live in this repository.
For a plain-language explanation of UKB storage, the clean datasets, each
dataset graph, and the exact reuse boundary, see the
[dataset, graph, and CRAG-reuse guide](docs/DATASET_GRAPH_PROVENANCE.md).

## Novelty and Positioning

The novelty is **not** the generic idea of using an MLP with precomputed graph
features, being the first non-message-passing graph model, or showing that
message passing can be simplified. Prior work on simplified or precomputed graph
propagation, MLP-style graph models, and structural sketches already shows that
learned message passing can sometimes be replaced or approximated by
precomputed graph information in node-classification and link-prediction
settings. This project makes no firstness claim for that general idea.

### A. Retrieval-specific separation of graph information from message passing

The central scientific question is specific to retrieval ranking:

> **Does graph-aware retrieval require learned neighborhood aggregation, or can
> the useful structural signal be exposed through fixed query-local summaries
> and consumed by a lightweight ranker?**

Unlike graph features defined once per node for a generic prediction task, the
important summaries here are conditioned on the frozen retrieval seeds and the
query-local retrieval state. They include:

- distance from retrieval seeds;
- path counts and connectivity to seeds;
- seed-conditioned PPR or diffusion;
- other query-local structural descriptors registered before final evaluation.

The learned SA-MLP consumes those fixed summaries but performs no learned
neighborhood aggregation.

### B. Explicit fairness decomposition

The four-model chain is a core methodological contribution:

```text
plain MLP → seed-only MLP → SA-MLP → seed-aware GNN
```

It isolates three quantities that a direct MLP/GNN benchmark would confound:

- `seed-only - plain` measures the frozen retrieval prior;
- `SA-MLP - seed-only` measures fixed structural information beyond that prior;
- `SA-MLP - seed-aware GNN` compares fixed summaries with learned message
  passing after both sides receive the same seed indicator.

This decomposition is scientifically more important than whether one model has
the highest rounded mean on a particular dataset. It establishes which
information differs between the models and prevents retrieval-seed membership
from being mistaken for graph reasoning.

### C. Broad retrieval evidence

The final study covers six retrieval and QA datasets, five paired optimizer
seeds, paired-query statistics, Holm correction, approximately matched
parameter counts, and explicit latency, GPU-memory, CPU-memory, preprocessing,
and cache accounting.

The important empirical observation is not that SA-MLP always wins. It is:

> **Once retrieval-seed information and query-local graph structure are made
> explicit, seed-aware GNN message passing provides surprisingly little
> additional R@5 effectiveness across the six evaluated datasets while
> incurring a consistent warm-cache reranking-latency and GPU-memory cost.**

The conclusion is bounded to the evaluated retrieval regimes, frozen candidate
pools, model families, and training protocol.

### D. Multi-hop result

MetaQA contradicts the simple hypothesis that deeper query hop count necessarily
makes learned message passing more valuable:

- at one hop, the seed-aware GNN is slightly ahead;
- at two hops, SA-MLP is slightly ahead;
- at three hops, SA-MLP is slightly ahead.

The current evidence therefore does not support “more hops implies that a GNN
is more necessary.” It also does not establish that fixed summaries are
sufficient for every multi-hop task.

### E. Systems tradeoff

SA-MLP and the selected GNNs are approximately parameter matched, so the systems
advantage is not explained by parameter-count reduction. The contribution is
moving learned neighborhood propagation out of the scorer and replacing it
with fixed structural computation plus a lightweight MLP ranker. The completed
latency measurement assumes reusable per-query caches; for a genuinely unseen
query, query-local summaries must instead be computed on demand.

SA-MLP trades:

- structural-feature preprocessing;
- CPU-accessible arrays and disk caches;

for:

- lower warm-cache candidate-reranking latency;
- substantially lower incremental GPU memory;
- a simpler learned inference path without adjacency or neighbor aggregation.

Both sides of this tradeoff are reported; preprocessing and storage are not
treated as free.

### F. Relationship to prior work

Simplified and precomputed propagation work shows that propagation can sometimes
be moved outside the learned loop. MLP-style graph methods show that message
passing is not always necessary for graph prediction. Structural-sketch methods
show that compact local summaries can support efficient graph prediction.

This project differs by studying retrieval ranking with query-local,
seed-conditioned structural summaries; explicitly separating the retrieval
prior, fixed structure, and learned propagation; and evaluating the resulting
accuracy–latency–memory tradeoff across multiple QA and KB retrieval regimes.
It does not claim novelty beyond those demonstrated distinctions. Shared CRAG
datasets and graphs remain experimental substrate, not part of the novelty
claim.

### G. Current novelty statement

> **Current novelty:** We study whether learned message passing is necessary for
> graph-aware retrieval by separating retrieval prior, fixed query-local
> structural information, and learned neighborhood aggregation under a
> controlled, approximately parameter-matched protocol. Across six retrieval
> benchmarks, fixed structural summaries recover nearly all of the retrieval
> effectiveness of seed-aware GNNs while substantially reducing warm-cache
> reranking compute and GPU memory.

### H. What is still needed for a stronger NeurIPS claim

The completed study is strong retrieval evidence, not yet a general theory of
message-passing utility. The main deferred steps are:

- causal topology perturbations;
- feature-quality perturbations;
- a predictor for when message passing helps enough to justify its cost;
- leave-one-dataset-out validation of that predictor;
- evaluation in broader non-QA graph domains.

These steps are needed to move from a controlled empirical retrieval study to a
deeper statement about when fixed structural summaries are sufficient and when
learned message passing is genuinely necessary. They remain future work; no
such experiment is currently scheduled.

## Scientific contract

Every primary comparison uses the same frozen node/query features, candidate
IDs and order, relevance labels, loss, negative pool, split, optimizer seed,
training budget, and validation-only checkpoint rule. Test is evaluated once.
The topology or its fixed summaries are the controlled additional information.

The final analysis reports five-seed mean and sample standard deviation,
paired-seed confidence intervals, two-stage paired query confidence intervals,
and Holm correction across the six datasets per contrast and metric. R@5 is
primary; R@1, R@20, MRR, and FullCov@20 are always reported. Parameters,
training time, warm-cache candidate-reranking latency, GPU allocation, CPU
memory, feature-precomputation time, and cache size are reported separately.
The completed latency table is not an end-to-end unseen-query benchmark.

All SA features are inference-safe functions of frozen retrieval seeds,
candidate IDs/order, graph topology, and registered numerical constants. Gold
documents, support labels, relevance positions, predictions, and learned model
state do not enter feature construction.

## Experimental progression

The project progressed through explicit frozen gates rather than one continuous
architecture search:

1. Parameter-matched plain MLPs were compared with GCN, GraphSAGE, GAT, and GIN
   under identical retrieval candidates, labels, losses, and seeds.
2. Offset-MLP and K-direction Offset operators tested whether lightweight
   embedding-space relational directions could replace neighborhood
   aggregation.
3. A five-seed confirmation established that the surprising plain-MLP R@5 wins
   on 2Wiki and MuSiQue survived optimizer variance.
4. A six-dataset study extended the matched plain-MLP comparison to 2Wiki,
   MuSiQue, WebQSP, HotpotQA, SQuAD, and MetaQA, selecting each GNN family by
   validation R@5 only.
5. That study showed a dataset-dependent boundary: two plain-MLP wins, three
   GNN wins, and one neutral dataset, with a consistent MLP systems advantage.
6. SA-MLP was developed to expose fixed global and query-local graph summaries
   to a non-message-passing scorer.
7. The one-seed screen revealed a fairness issue: distance zero exposed frozen
   retriever-seed membership to SA-MLP but not to the original GNN comparator.
8. The final frozen confirmation therefore compared four distinct models on all
   six datasets and five paired seeds: plain MLP, seed-only MLP, unchanged
   SA-MLP, and seed-aware validation-selected GNN.

Historical details and negative gates are retained in
[the Offset screen](docs/OFFSET_SCREEN_RESULTS.md),
[plain-MLP confirmation](docs/CONFIRMATION_RESULTS.md),
[coverage-variant result](docs/COVERAGE_VARIANT_RESULTS.md),
[six-dataset result](docs/SIX_DATASET_RESULTS.md), and
[SA-MLP screen](docs/SA_MLP_SCREEN_RESULTS.md).

## Final fairness comparison

The four-model interpretation contract prevents the retrieval prior, fixed
graph computation, and learned message passing from being collapsed into one
comparison:

| Contrast | Interpretation |
|---|---|
| `seed-only MLP - plain MLP` | Value of the frozen retrieval prior |
| `SA-MLP - seed-only MLP` | Value of fixed graph computation beyond that prior |
| `SA-MLP - seed-aware GNN` | Fixed structural summaries versus learned message passing |

### Final Recall@5

Values are five-seed means from the frozen confirmation result. The difference
is SA-MLP minus seed-aware GNN in percentage points.

| Dataset | SA-MLP R@5 | Seed-aware GNN R@5 | SA − GNN |
|---|---:|---:|---:|
| 2Wiki | 68.40 | 69.85 | -1.44 |
| MuSiQue | 80.28 | 81.24 | -0.96 |
| WebQSP | 33.37 | 33.09 | +0.27 |
| HotpotQA | 77.13 | 77.66 | -0.53 |
| SQuAD | 89.23 | 89.33 | -0.10 |
| MetaQA | 30.11 | 30.13 | -0.02 |

Across all six datasets, SA-MLP remains within 1.44 R@5 points of the
seed-aware GNN. This compact range is not a universal SA-MLP win:

- 2Wiki has a statistically supported 1.44-point GNN advantage and fails the
  preregistered one-point non-inferiority margin;
- MuSiQue's 0.96-point mean deficit has intervals that extend beyond the margin,
  so non-inferiority is not certified;
- WebQSP is numerically +0.27 for SA-MLP but its paired-query interval is wide,
  so it remains inconclusive;
- HotpotQA, SQuAD, and MetaQA are within the one-point margin under both the
  paired-seed and paired-query intervals, although HotpotQA retains a small,
  statistically supported R@5 advantage for the GNN.

The preregistered substitution gate considered the three datasets on which the
earlier plain MLP lost to the GNN. Requiring both registered intervals to clear
the margin, SA-MLP is non-inferior on MetaQA and HotpotQA (2/3, gate passed),
while WebQSP is query-level inconclusive. Full metric vectors, intervals, and
Holm-adjusted tests are in the
[confirmation result](docs/SA_MLP_CONFIRMATION_RESULTS.md).

### Structural gains are not seed leakage

Seed membership was identified before confirmation as an information asymmetry.
The architecture was frozen, a seed-only MLP was added, and the selected GNN
received the same binary seed feature. The resulting `SA-MLP - seed-only` R@5
effects include:

| Dataset | SA − seed-only R@5 |
|---|---:|
| 2Wiki | +2.58 |
| WebQSP | +4.11 |
| HotpotQA | +3.70 |
| MetaQA | +6.86 |

These gains are supported by the frozen paired statistical analysis. MuSiQue
(+0.20) and SQuAD (-0.08) show little additional R@5 signal from the full
structural package, which is useful evidence that graph value is
regime-dependent.

The defensible conclusion is that SA-MLP is not merely copying the initial
retriever preference: fixed distance, path/connectivity, and query-seeded
diffusion/PPR-style summaries add measurable information beyond seed membership
in several regimes.

### MetaQA hop result

The fairness-controlled MetaQA result does not show a growing advantage for
message passing as query hop count increases:

| Hop | SA-MLP R@5 | Seed-aware GNN R@5 | SA − GNN |
|---:|---:|---:|---:|
| 1 | 76.06 | 76.97 | -0.91 |
| 2 | 16.86 | 16.58 | +0.28 |
| 3 | 11.90 | 11.61 | +0.29 |

Increasing reasoning depth does not automatically increase the value of learned
message passing in this substrate. SA-MLP is slightly higher at two and three
hops, so hop count alone cannot explain the crossover. This is not proof that
message passing is unnecessary for every multi-hop reasoning problem.

## Systems result

Trainable parameter counts are approximately matched: the plain MLP has about
205K parameters, and SA-MLP/seed-aware GNN configurations are about 209K–214K.
There is no 2–4× parameter-reduction result. The efficiency contribution comes
from avoiding learned message propagation inside the cached scoring path.

| Dataset | GNN/SA cached latency | SA GPU MiB | GNN GPU MiB | SA feature cache GiB |
|---|---:|---:|---:|---:|
| 2Wiki | 3.7× | 53.0 | 143.3 | 0.102 |
| MuSiQue | 3.5× | 50.0 | 301.2 | 0.124 |
| WebQSP | 4.6× | 51.5 | 298.8 | 0.030 |
| HotpotQA | 2.9× | 53.1 | 2400.7 | 0.650 |
| SQuAD | 7.1× | 49.3 | 2467.5 | 0.776 |
| MetaQA | 2.5× | 55.8 | 205.2 | 2.835 |

SA-MLP is approximately 2.5–7.1× faster in the completed warm-cache
candidate-reranking benchmark. The GPU-memory difference is especially large
on HotpotQA (about 53 MiB versus 2,401 MiB) and SQuAD (about 49 MiB versus
2,468 MiB).

This is not free computation. The completed benchmark precomputes deterministic
structural features into CPU-accessible arrays and reusable disk caches.
Measured bulk preprocessing takes 9.3–20.5 seconds for the frozen datasets, and
the largest feature cache is 2.835 GiB on MetaQA. The systems interpretation is:

> **SA-MLP trades reusable offline structural preprocessing and storage for
> substantially cheaper cached learned inference.**

This ratio must not be presented as an end-to-end production-query speedup.
Both methods currently read a prepacked candidate-induced topology, and SA-MLP
also reads precomputed query-local features. A future unseen-query benchmark
must charge both methods for candidate graph induction and charge SA-MLP for
on-demand distance/path/PPR computation. The exact future protocol is recorded
in [RRF fusion and unseen-query systems evaluation](docs/RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md).

## What features mattered

The frozen mechanism screen gives a useful but incomplete explanation:

- global degree, PageRank, coreness, clustering, and hub-style descriptors alone
  were not useful and could substantially hurt retrieval;
- query-local structural features carried the main signal;
- useful families included distance from frozen retrieval seeds, path counts and
  connectivity to those seeds, and query-seeded diffusion/PPR-style summaries;
- WebQSP benefited from the combined interaction and query-local package rather
  than either family alone.

These observations identify useful feature families. They do not establish a
general causal theory or prove that the same summaries are sufficient in every
graph domain.

## Negative results and lessons

Negative results materially shaped the final research question and remain part
of the project record:

- Offset-MLP did not generally outperform the GNN comparators.
- Offset showed an early-rank precision versus multi-answer-coverage tradeoff.
- Offset's coverage deficit worsened as answer multiplicity increased.
- The preregistered K-direction, permutation-invariant set-assignment remedy
  failed on both 2Wiki and MuSiQue; no second remedy was tried.
- Validation selected the full roughly 205K–221K configurations. Smaller
  roughly 50K/100K MLPs lost too much effectiveness, so there is no 2–4×
  parameter-reduction claim.
- Global structural descriptors alone did not produce SA-MLP's gains.
- Hop count alone does not predict GNN utility.
- The plain MLP does not dominate across datasets, and SA-MLP still has a clear
  one-point-margin failure on 2Wiki.

## Deferred Future Work

Everything in this section is deferred. No experiment below is currently
scheduled or authorized by the frozen confirmation protocol. Any resumption
requires a separate preregistration that does not tune or filter the completed
test results.

### A. Causal topology perturbations

Use degree-preserving rewiring and controlled edge corruption to test whether
learned message passing and fixed summaries degrade differently as topology
quality falls.

### B. Feature corruption

Systematically weaken semantic node features to identify regimes where learned
message passing becomes genuinely valuable rather than merely redundant.

### C. Predicting the crossover

Develop a dataset-independent or query-level predictor from neighborhood
coherence, degree/hub exposure, local structural entropy, seed-to-candidate
connectivity, path redundancy, candidate multiplicity, and graph-versus-semantic
agreement. The target is whether message passing will improve retrieval enough
to justify its systems cost.

### D. Leave-one-dataset-out generalization

Fit the crossover predictor on N−1 datasets and evaluate whether it predicts the
held-out graph regime without access to that dataset's test outcomes.

### E. Broader graph domains

Validate beyond QA and retrieval graphs on graph families with substantially
different topology, feature quality, edge semantics, and supervision.

### F. Stronger GNN families

Compare modern scalable, sparse, or query-conditioned message-passing methods
while preserving identical candidates, features, supervision, validation
budgets, and systems accounting.

### G. Structural-summary design

Study whether fixed summaries can be compressed, selected, or made more compact
without turning the learned scorer into message passing.

### H. Systems scaling

Extend offline-versus-online accounting to cache compression, dynamic-graph
updates, feature invalidation, and amortization across query volume. Run a true
unseen-query benchmark from raw query text that includes query encoding, Dense
and SPLADE retrieval, candidate fusion, graph induction, on-demand SA features,
model inference, and top-K selection. Preserve the existing cached-reranker
number as a separate operator-cost diagnostic.

### I. Dense/SPLADE reciprocal-rank fusion

Add a locked equal-RRF baseline over the existing Dense and SPLADE top-200
rankings. Standard RRF uses rank positions, so the frozen ID arrays are
sufficient even though raw score arrays were not exported. Test validation-only
weighted RRF separately, and give any RRF scalar or RRF-derived seed set
identically to SA-MLP and seed-aware GNN controls. RRF over the unchanged union
can improve in-pool ordering but cannot improve candidate ceiling.

### J. Theory and mechanism

Develop a principled explanation for when query-local structural sufficient
statistics can substitute for iterative neighborhood aggregation. This is the
main missing element before treating the work as mature for a NeurIPS
submission.

## Repository guide

```text
configs/                         frozen and historical experiment contracts
data/                            data contract; generated tensors are ignored
docs/NEURIPS_RESEARCH_PLAN.md    historical roadmap and deferred research plan
docs/DATASET_GRAPH_PROVENANCE.md datasets, graphs, UKB storage, and CRAG reuse
docs/RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md RRF and unseen-query timing plan
docs/SA_MLP_CONFIRMATION_PROTOCOL.md final frozen fairness protocol
docs/SA_MLP_CONFIRMATION_RESULTS.md complete six-dataset result and stopping point
docs/SA_FEATURE_LEAKAGE_AUDIT.md label-free fixed-feature audit
docs/LEGACY_CANDIDATE_COMPATIBILITY.md legacy candidate-equivalence proof
docs/SIX_DATASET_RESULTS.md      earlier plain-MLP/GNN boundary
docs/CONFIRMATION_RESULTS.md     five-seed plain/Offset confirmation
docs/COVERAGE_VARIANT_RESULTS.md failed preregistered Offset remedy
legacy/crag_snapshot/            provenance snapshots; not production code
src/mp_retrieval/                standalone research implementation
tests/                           unit, parity, and contract tests
```

The environment can be validated without launching research experiments:

```bash
python -m pip install -e ".[graph,dev]"
pytest
```

Compute credentials are resolved at runtime from an ignored local file or
environment variables. Private CRAG credentials must never be copied into this
repository. Experimental launch commands remain in the historical protocol
record for reproducibility, but the project status above takes precedence: no
new runs are currently scheduled.
