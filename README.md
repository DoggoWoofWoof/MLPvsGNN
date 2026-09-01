# When Does Graph-Aware Retrieval Need Message Passing?

**Working subtitle:** *Fixed Structural Summaries versus Learned Neighborhood
Aggregation*

> **Target paper identity:** We characterize when graph-aware retrieval
> benefits from learned neighborhood propagation and when fixed query-local
> structural computation is sufficient.

## Current status

**Fairness-confirmed six-dataset study: COMPLETE. Package A1/A2/A3 controls:
COMPLETE. Candidate-generation headroom diagnostic: COMPLETE. Packages B/C/E1:
PARTIAL, INTEGRITY-VERIFIED, AND RESUMED. Package D: FROZEN AND GATED ON C.
Package F: SEALED.** The sealed QLS/GNN checkpoint remains unchanged and no
partial seed outcome has been interpreted.

The workspace spend limit that blocked GPU allocation on 2026-08-31 has lifted.
B, C, and E1 were resumed on 2026-09-01 with every volume checkpoint preserved.
The exact completion matrices, remaining work, and safe resume order are in the
[execution-status checkpoint](docs/EXPERIMENT_EXECUTION_STATUS.md).

Every reported metric now has a companion ceiling. The
[headroom diagnostic](docs/CANDIDATE_HEADROOM_RESULTS.md) separates upstream
candidate-generation failure from the reporting cut-off and from reranking, so
QLS-MLP and the seed-aware GNN are neither credited nor penalized for gold
evidence that was never in their candidate set. It reorders the cross-dataset
reading: MetaQA attains 92% of what its candidates allowed and is not a
modelling failure, while WebQSP attains only 74% and holds the most unexploited
reranking headroom.

| Established | Not yet established |
|---|---|
| Four-level plain/seed/QLS/GNN decomposition | Uncached post-retrieval speedup |
| Five paired seeds on all six datasets | Edge-provenance/native-versus-kNN mechanism |
| QLS nearly recovers seed-aware-GNN R@5 | Topology/feature robustness crossover |
| Warm-cache learned-inference advantage | Message-passing utility predictor |
| Leakage-safe data/graph audit | Untouched external confirmation |
| Dense/SPLADE/equal/weighted-RRF controls | Candidate-pool budget sweep |
| Fixed distance/PPR/path/fusion controls | Upstream seed-quality robustness |
| 19-parameter linear rank+structure control | Typed-edge ablation |
| Candidate ceilings for every reported metric | Whether expansion lifts the ceiling (Paper 2) |

This is a standalone research repository about the value and cost of graph
message passing for retrieval. It began with the question:

> **Do MLPs outperform GNN message passing for retrieval?**

The evidence led to the paper's primary scientific question:

> **Once retrieval has already happened, how much learned message passing is
> still necessary to rank graph-aware evidence?**

The final fixed-structure model is publication-facing **QLS-MLP**
(**Query-Local Structure MLP**). It is graph-aware and
**non-message-passing**. It uses deterministic structural features computed from
the frozen graph and retrieval seeds, but its learned forward pass never
aggregates neighbor embeddings or receives adjacency. It is therefore **not
topology-free**. Its trainable parameter count is approximately matched to the
GNN comparators; fewer parameters are not a contribution of this work.
The separate 19-parameter A3 model is deliberately a low-capacity diagnostic
control, not a replacement name or parameter claim for QLS-MLP.

The frozen implementation key remains `sa_mlp`, and the sealed SA-named files,
tags, configurations, and hashes are intentionally unchanged. `SA-MLP` is
already the name of a published TMLR method; all new paper prose therefore uses
QLS-MLP. See the
[terminology and positioning note](docs/TERMINOLOGY_AND_POSITIONING.md) for the
publication-to-artifact mapping and the revised related-work boundary.

## Current Research Finding

> **Across six retrieval benchmarks, a non-message-passing MLP supplied with
> fixed query-local graph summaries recovers nearly all of the Recall@5
> effectiveness of parameter-matched seed-aware GNNs, remaining within 1.44
> points on every dataset while substantially reducing cached online learned-
> inference cost and GPU memory.
> Controlled seed-only ablations show that the structural gain is not explained
> solely by retriever-seed membership.**

The broader conceptual result is:

> **Graph structure can be useful for retrieval without requiring learned
> message passing.**

This does not mean that MLPs universally beat GNNs. GNNs retain small but
statistically supported advantages in some datasets and metrics; WebQSP is
numerically slightly positive for QLS-MLP; and several regimes are effectively
at parity. The central finding is that learned message passing provides
surprisingly little incremental retrieval effectiveness once useful
query-local structural information is explicitly exposed.

The project does **not** claim that graphs are useless, QLS-MLP is topology-free,
QLS-MLP has fewer parameters, or message passing is never required.

The completed P0 controls sharpen this boundary. Validation-selected RRF improves on
the stronger single ranker by 0.00–2.64 R@5 points. Fixed structural summaries
alone add +5.20 points over RRF on WebQSP and +4.41 on MetaQA, while locked
RRF+PPR adds +1.01 on HotpotQA. Yet the best training-free control still trails
QLS-MLP by 3.89–17.97 points on HotpotQA, MuSiQue, MetaQA, and WebQSP. Simple
rank fusion or a single hand-built structural rule therefore does not explain
the complete QLS result. The 19-parameter A3 linear control recovers part of
this gap—most clearly on WebQSP, MetaQA, and HotpotQA—but still trails QLS by
11.77, 11.16, 5.63, and 2.55 R@5 points on MuSiQue, WebQSP, MetaQA, and
HotpotQA, respectively. It matches QLS within one point only on 2Wiki and
SQuAD. This isolates nonlinear semantic/structural interaction as important in
four datasets without implying that message passing itself is necessary.

## System boundary

This repository is a **post-retrieval graph-aware candidate-ranking study**:

```text
query embedding + Dense ranked IDs + SPLADE ranked IDs + frozen corpus graph
                                 |
                                 v
                  candidate fusion and induced graph
                                 |
               +-----------------+-----------------+
               |                                   |
     fixed query-local summaries          learned message passing
            -> QLS-MLP                     -> seed-aware GNN
               |                                   |
               +-----------------+-----------------+
                                 v
                         candidate ranking
```

It is not C-RAG, a UKB runtime, a raw-text search engine, or a full RAG system.

## Frozen checkpoint

The authoritative checkpoint consists of:

- frozen protocol: tag `sa-mlp-confirmation-protocol-v1`, commit `00951ec`;
- legacy-candidate compatibility proof: tag
  `sa-mlp-confirmation-compat-v1`, commit `63b8f85`;
- frozen analysis compiler: tag `sa-mlp-confirmation-analysis-v1`, commit
  `7561049`;
- completed result and stopping decision: tag
  `sa-mlp-confirmation-results-v1`, commit `eca5cbd`;
- protocol document: [QLS-MLP confirmation protocol](docs/SA_MLP_CONFIRMATION_PROTOCOL.md);
- complete five-seed tables: [QLS-MLP confirmation results](docs/SA_MLP_CONFIRMATION_RESULTS.md);
- feature-safety evidence: [QLS feature leakage audit](docs/SA_FEATURE_LEAKAGE_AUDIT.md);
- legacy candidate proof: [candidate compatibility audit](docs/LEGACY_CANDIDATE_COMPATIBILITY.md).

The earlier general comparison is separately frozen at
`six-dataset-protocol-v1` (`f012699`) and documented in
[the six-dataset results](docs/SIX_DATASET_RESULTS.md). Historical protocol tags
remain part of the record and must not be moved or deleted.

The separate Package A resumption is versioned by:

- corrected rank controls: `p0-rank-controls-results-v2`, with
  [A1 results](docs/P0_RANK_CONTROLS_RESULTS.md);
- fixed structural controls: `p0-fixed-structural-controls-results-v1`, with
  [A2 results](docs/P0_FIXED_STRUCTURAL_CONTROLS_RESULTS.md);
- learned linear control: `p0-linear-rank-structure-results-v1`, with
  [A3 results](docs/P0_LINEAR_RANK_STRUCTURE_RESULTS.md); and
- explicit protocol/correction notes in the
  [A1 protocol](docs/P0_RANK_CONTROLS_PROTOCOL.md),
  [A1 MRR correction](docs/P0_RANK_CONTROLS_METRIC_CORRECTION.md), and
  [A2 protocol](docs/P0_FIXED_STRUCTURAL_CONTROLS_PROTOCOL.md), plus the
  [A3 protocol](docs/P0_LINEAR_RANK_STRUCTURE_PROTOCOL.md) and
  [bias amendment](docs/P0_LINEAR_CONTROL_BIAS_AMENDMENT.md).

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
settings. This includes SGC, SIGN, Graph-MLP, BUDDY, the existing published
[SA-MLP](https://openreview.net/forum?id=MZ2kKZc8m7), and the contemporaneous
[RTA](https://arxiv.org/abs/2608.26732) work. This project makes no firstness
claim for that general idea.

### A. Retrieval-specific separation of graph information from message passing

The central scientific question is specific to retrieval ranking:

> **For graph-aware candidate retrieval, what portion of the benefit attributed
> to GNN message passing comes from upstream retrieval priors and explicit
> query-local structural statistics?**

Unlike graph features defined once per node for a generic prediction task, the
important summaries here are conditioned on the frozen retrieval seeds and the
query-local retrieval state. They include:

- distance from retrieval seeds;
- path counts and connectivity to seeds;
- seed-conditioned PPR or diffusion;
- other query-local structural descriptors registered before final evaluation.

The learned QLS-MLP consumes those fixed summaries but performs no learned
neighborhood aggregation.

### B. Explicit fairness decomposition

The four-model chain is a core methodological contribution:

```text
Plain MLP
    ↓
Seed-only MLP
    ↓
QLS-MLP (fixed query-local structure)
    ↓
Seed-aware GNN (learned message passing)
```

Conceptually, this adds one information source at a time:

```text
semantic embedding signal
         ↓
upstream retrieval prior
         ↓
explicit fixed graph structure
         ↓
learned message passing
```

It isolates three quantities that a direct MLP/GNN benchmark would confound:

- `seed-only - plain` measures the frozen retrieval prior;
- `QLS-MLP - seed-only` measures fixed structural information beyond that prior;
- `QLS-MLP - seed-aware GNN` compares fixed summaries with learned message
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

The important empirical observation is not that QLS-MLP always wins. It is:

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
- at two hops, QLS-MLP is slightly ahead;
- at three hops, QLS-MLP is slightly ahead.

The current evidence therefore does not support “more hops implies that a GNN
is more necessary.” It also does not establish that fixed summaries are
sufficient for every multi-hop task.

### E. Systems tradeoff

QLS-MLP and the selected GNNs are approximately parameter matched, so the systems
advantage is not explained by parameter-count reduction. The contribution is
moving learned neighborhood propagation out of the scorer and replacing it
with fixed structural computation plus a lightweight MLP ranker. The completed
latency measurement assumes reusable per-query caches; for a genuinely unseen
query, query-local summaries must instead be computed on demand.

QLS-MLP trades:

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

The closest new overlap is RTA, *Rethinking Message Passing as Retrieval for
Text-Attributed Graph Learning*. The distinction is precise:

| Dimension | RTA | This project |
|---|---|---|
| Task | Node prediction/text-attributed graph learning | Query-candidate retrieval ranking |
| Starting object | Graph node | External query |
| Context retrieval | Builds its own retrieval context | Dense/SPLADE already produced candidates |
| Structural signal | Node-conditioned PPR | Retrieval-seed-conditioned distance/path/PPR |
| Labels as inference context | Possible | Prohibited |
| Neighbor representation aggregation | Yes | QLS consumes fixed scalar summaries |
| Main question | Can retrieval replace message passing? | What does message passing add after retrieval? |

The different conditioning is central:

```text
RTA: PPR(node u -> graph)
QLS: PPR(retrieval seeds S(q) -> candidate d)
```

This project does not claim to invent PPR-based structural retrieval. Its core
distinction is the four-level retrieval-ranking decomposition.

### G. Mechanistic hypotheses

> **H1 — Structural compressibility:** QLS-MLP should approach or match learned
> message passing when the graph information useful for ranking a candidate can
> be compressed into seed membership, seed distance, path multiplicity or
> connectivity, and seed-conditioned diffusion.

> **H2 — Rich-content requirement:** GNNs should gain an advantage when ranking
> depends on richer neighbor content, interactions, typed/compositional
> relations, or higher-order information that the fixed summaries do not
> represent adequately.

These are mechanistic hypotheses, not established theorems. The goal is to
predict when fixed structural computation is sufficient and when learned
message passing is worth its additional cost.

### H. Current novelty statement

> **Current novelty:** We quantify, for graph-aware candidate retrieval, how
> much apparent GNN benefit is explained by the frozen upstream retrieval prior,
> how much is recovered by explicit query-local structural statistics, and what
> incremental value remains for learned neighborhood propagation under a
> controlled, approximately parameter-matched protocol.

### I. Target NeurIPS claim and required evidence

The completed study is strong retrieval evidence, not yet a general theory of
message-passing utility. The intended claim is explicitly a **target, not an
established result**:

> **We characterize measurable retrieval and graph regimes in which fixed
> query-local structural computation is sufficient and regimes in which
> learned neighborhood aggregation provides additional value.**

The six required packages are:

| Package | Deferred experiment | Purpose |
|---|---|---|
| A | Semantic versus structural controls | Locate which information recovers GNN benefit |
| B | Native/title/KB versus kNN edge provenance | Determine whether benefit is relational or recycled embedding geometry |
| C | Candidate/context budgets 50/100/200/400 | Locate structural-context saturation and cost |
| D | Uncached post-retrieval systems evaluation | Measure true online ranker cost from unseen embeddings/rankings |
| E | Robustness phase diagram and utility predictor | Explain and predict the QLS/GNN crossover |
| F | Fresh untouched retrieval-plus-graph confirmation | Confirm frozen hypotheses and predictor externally |

Package F is deliberately last: A–E must be frozen before its test outcomes are
seen. The desired main-track spine is **empirical decomposition + causal
perturbation + crossover predictor + systems tradeoff + external confirmation**.

These steps are needed to move from a controlled empirical retrieval study to a
deeper statement about when fixed structural summaries are sufficient and when
learned message passing is genuinely necessary. Packages B and C and the
validation-only E1 screen are frozen, partially persisted, and resumed as of
2026-09-01. Package D is committed but cannot launch until the budget-400
checkpoints from C are complete. Package F remains deliberately unopened.

For the NeurIPS main track, the phase diagram/predictor or theory is the likely
gate. A separate Evaluations & Datasets route could instead center a documented
**MPR-Bench** protocol showing how retriever-prior, graph-information, and
learned-propagation confounds change conclusions. The two routes must be scoped
explicitly before resuming experiments.

## Scientific contract

Every primary comparison uses the same frozen node/query features, candidate
IDs and order, relevance labels, loss, negative pool, split, optimizer seed,
training budget, and validation-only checkpoint rule. Each sealed protocol
evaluates its final seeds on test once. Because the broader project progressed
through sequential screens on the same benchmark families, a future headline
confirmation requires a fresh untouched external holdout. The topology or its
fixed summaries are the controlled additional information.

The final analysis reports five-seed mean and sample standard deviation,
paired-seed confidence intervals, two-stage paired query confidence intervals,
and Holm correction across the six datasets per contrast and metric. R@5 is
primary; R@1, R@20, MRR, and FullCov@20 are always reported. Parameters,
training time, warm-cache candidate-reranking latency, GPU allocation, CPU
memory, feature-precomputation time, and cache size are reported separately.
The completed latency table is not an uncached post-retrieval benchmark for
unseen query embeddings.

All QLS features are inference-safe functions of frozen retrieval seeds,
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
6. QLS-MLP was developed to expose fixed global and query-local graph summaries
   to a non-message-passing scorer.
7. The one-seed screen revealed a fairness issue: distance zero exposed frozen
   retriever-seed membership to QLS-MLP but not to the original GNN comparator.
8. The final frozen confirmation therefore compared four distinct models on all
   six datasets and five paired seeds: plain MLP, seed-only MLP, unchanged
   QLS-MLP, and seed-aware validation-selected GNN.

Historical details and negative gates are retained in
[the Offset screen](docs/OFFSET_SCREEN_RESULTS.md),
[plain-MLP confirmation](docs/CONFIRMATION_RESULTS.md),
[coverage-variant result](docs/COVERAGE_VARIANT_RESULTS.md),
[six-dataset result](docs/SIX_DATASET_RESULTS.md), and
[QLS-MLP screen](docs/SA_MLP_SCREEN_RESULTS.md).

## Final fairness comparison

The four-model interpretation contract prevents the retrieval prior, fixed
graph computation, and learned message passing from being collapsed into one
comparison:

| Contrast | Interpretation |
|---|---|
| `seed-only MLP - plain MLP` | Value of the frozen retrieval prior |
| `QLS-MLP - seed-only MLP` | Value of fixed graph computation beyond that prior |
| `QLS-MLP - seed-aware GNN` | Fixed structural summaries versus learned message passing |

### Final Recall@5

Values are five-seed means from the frozen confirmation result. The difference
is QLS-MLP minus seed-aware GNN in percentage points.

| Dataset | QLS-MLP R@5 | Seed-aware GNN R@5 | QLS − GNN |
|---|---:|---:|---:|
| 2Wiki | 68.40 | 69.85 | -1.44 |
| MuSiQue | 80.28 | 81.24 | -0.96 |
| WebQSP | 33.37 | 33.09 | +0.27 |
| HotpotQA | 77.13 | 77.66 | -0.53 |
| SQuAD | 89.23 | 89.33 | -0.10 |
| MetaQA | 30.11 | 30.13 | -0.02 |

Across all six datasets, QLS-MLP remains within 1.44 R@5 points of the
seed-aware GNN. This compact range is not a universal QLS-MLP win:

- 2Wiki has a statistically supported 1.44-point GNN advantage and fails the
  preregistered one-point non-inferiority margin;
- MuSiQue's 0.96-point mean deficit has intervals that extend beyond the margin,
  so non-inferiority is not certified;
- WebQSP is numerically +0.27 for QLS-MLP but its paired-query interval is wide,
  so it remains inconclusive;
- HotpotQA, SQuAD, and MetaQA are within the one-point margin under both the
  paired-seed and paired-query intervals, although HotpotQA retains a small,
  statistically supported R@5 advantage for the GNN.

The preregistered substitution gate considered the three datasets on which the
earlier plain MLP lost to the GNN. Requiring both registered intervals to clear
the margin, QLS-MLP is non-inferior on MetaQA and HotpotQA (2/3, gate passed),
while WebQSP is query-level inconclusive. Full metric vectors, intervals, and
Holm-adjusted tests are in the
[confirmation result](docs/SA_MLP_CONFIRMATION_RESULTS.md).

### Structural gains are not seed leakage

Seed membership was identified before confirmation as an information asymmetry.
The architecture was frozen, a seed-only MLP was added, and the selected GNN
received the same binary seed feature. The resulting `QLS-MLP - seed-only` R@5
effects include:

| Dataset | QLS − seed-only R@5 |
|---|---:|
| 2Wiki | +2.58 |
| WebQSP | +4.11 |
| HotpotQA | +3.70 |
| MetaQA | +6.86 |

These gains are supported by the frozen paired statistical analysis. MuSiQue
(+0.20) and SQuAD (-0.08) show little additional R@5 signal from the full
structural package, which is useful evidence that graph value is
regime-dependent.

The defensible conclusion is that QLS-MLP is not merely copying the initial
retriever preference: fixed distance, path/connectivity, and query-seeded
diffusion/PPR-style summaries add measurable information beyond seed membership
in several regimes.

### MetaQA hop result

The fairness-controlled MetaQA result does not show a growing advantage for
message passing as query hop count increases:

| Hop | QLS-MLP R@5 | Seed-aware GNN R@5 | QLS − GNN |
|---:|---:|---:|---:|
| 1 | 76.06 | 76.97 | -0.91 |
| 2 | 16.86 | 16.58 | +0.28 |
| 3 | 11.90 | 11.61 | +0.29 |

Increasing reasoning depth does not automatically increase the value of learned
message passing in this substrate. QLS-MLP is slightly higher at two and three
hops, so hop count alone cannot explain the crossover. This is not proof that
message passing is unnecessary for every multi-hop reasoning problem.

## Systems result

Trainable parameter counts are approximately matched: the plain MLP has about
205K parameters, and QLS-MLP/seed-aware GNN configurations are about 209K–214K.
There is no 2–4× parameter-reduction result. The efficiency contribution comes
from avoiding learned message propagation inside the cached scoring path.

| Dataset | GNN/QLS cached latency | QLS GPU MiB | GNN GPU MiB | QLS feature cache GiB |
|---|---:|---:|---:|---:|
| 2Wiki | 3.7× | 53.0 | 143.3 | 0.102 |
| MuSiQue | 3.5× | 50.0 | 301.2 | 0.124 |
| WebQSP | 4.6× | 51.5 | 298.8 | 0.030 |
| HotpotQA | 2.9× | 53.1 | 2400.7 | 0.650 |
| SQuAD | 7.1× | 49.3 | 2467.5 | 0.776 |
| MetaQA | 2.5× | 55.8 | 205.2 | 2.835 |

QLS-MLP is approximately 2.5–7.1× faster in the completed warm-cache
candidate-reranking benchmark. The GPU-memory difference is especially large
on HotpotQA (about 53 MiB versus 2,401 MiB) and SQuAD (about 49 MiB versus
2,468 MiB).

This is not free computation. The completed benchmark precomputes deterministic
structural features into CPU-accessible arrays and reusable disk caches.
Measured bulk preprocessing takes 9.3–20.5 seconds for the frozen datasets, and
the largest feature cache is 2.835 GiB on MetaQA. The systems interpretation is:

> **QLS-MLP trades reusable offline structural preprocessing and storage for
> substantially cheaper cached learned inference.**

This ratio must not be presented as an uncached post-retrieval speedup.
Both methods currently read a prepacked candidate-induced topology, and QLS-MLP
also reads precomputed query-local features. A future unseen-embedding benchmark
must begin from an upstream query embedding plus Dense/SPLADE ranked IDs, charge
both methods for candidate graph induction, and charge QLS-MLP for on-demand
distance/path/PPR computation. Query encoding and initial retrieval remain
outside this paper's scope. The exact future protocol is recorded in
[RRF fusion and unseen-embedding systems evaluation](docs/RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md).

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
- Global structural descriptors alone did not produce QLS-MLP's gains.
- Hop count alone does not predict GNN utility.
- The plain MLP does not dominate across datasets, and QLS-MLP still has a clear
  one-point-margin failure on 2Wiki.

## Prioritized Remaining Work

The sealed confirmation remains closed to retuning. Package A1/A2/A3 below were
resumed only under new preregistered protocols and separate outputs. Every
remaining experiment still requires its own freeze and must not tune or filter
the completed test results.

The canonical details and priority order are in the
[paper-readiness and real-world audit](docs/PAPER_READINESS_AND_REAL_WORLD_FUTURE_WORK.md).

### Package A — Semantic versus structural decomposition

**A1, A2, and A3 are complete and closed.** Dense, SPLADE,
equal/validation-weighted RRF, structural-only PPR/distance/path summaries,
locked RRF+structure rules, and the 19-parameter linear rank+structure control
are reported in the linked P0 result documents alongside the sealed QLS-MLP
and seed-aware-GNN references. This ladder shows that fixed structure and
learned weighting help, but linear capacity does not explain QLS on four of six
datasets. Do not tune Package A further against these tests.

### Package B — Edge provenance (mandatory)

**Protocol frozen; partial runs integrity-verified and resumed 2026-09-01.**
Verified sidecars reconstruct
native/title/KB-only, embedding-kNN-only, simple-A, and union graphs. The exact
sealed A multigraph is reused, and simple A is a mandatory duplicate-
normalization control.

### Package C — Structural-context budget

**Protocol frozen; partial runs integrity-verified and resumed 2026-09-01.**
Equal-RRF budgets 50/100/200/400 jointly record candidate ceiling, retrieval
metrics, induced graph context, QLS/GNN compute, and checkpoints for Package D.
No budget is selected on test. Budget effects must be read against the
per-budget ceiling: that ceiling rises monotonically with budget on every
dataset, so part of any raw gain across the sweep is the ceiling moving rather
than the model reranking better.

### Package D — Online systems evaluation

**Protocol frozen; execution gated on Package C budget 400.** The uncached path
starts from a held-out query embedding and Dense/SPLADE rankings and charges
fusion, graph induction, method-specific computation, transfer, scoring, and
top-K. It reports batch 1/16 percentiles, throughput, method-level memory,
storage, and cold start. The existing 2.49–7.08x result remains warm-cache only.

### Package E — Robustness, crossover, and utility prediction

**E1 validation-only screen partial, integrity-verified, and resumed
2026-09-01.** Degree-preserving rewiring, random-edge
addition, hub injection, and raw-feature masking are screened with one seed on
validation only. A locked rule chooses crossover brackets. Five-seed test
confirmation cannot launch until those rates are generated, reviewed,
committed, and tagged. Predictor construction remains prohibited until the
confirmation establishes reproducible help/neutral/harm regions.

### Package F — Fresh untouched confirmation

After A–E and their hypotheses are frozen, evaluate once on unseen query
embeddings, upstream ranked candidates, relevance labels, and native or
preregistered label-free topology. NQ, MS MARCO, or BEIR alone is insufficient
without a graph contract.

### Future theory and optional work

The theoretical question is when fixed query-local summaries preserve the
ranking information obtainable by message passing. If relevance depends mainly
on seed membership, distance, reachable seeds, path multiplicity, diffusion,
and connectivity, fixed summaries may be sufficient. Rich neighbor semantics,
ordered or typed composition, and higher-order content transformations may
favor learned propagation. This is a mechanistic hypothesis, not a theorem.

Downstream QA, deeper formal theory, and additional GNN families remain
optional and must not displace the six packages.

## Repository guide

```text
configs/                         frozen and historical experiment contracts
data/                            data contract; generated tensors are ignored
docs/NEURIPS_RESEARCH_PLAN.md    historical roadmap and deferred research plan
docs/DATASET_GRAPH_PROVENANCE.md datasets, graphs, UKB storage, and CRAG reuse
docs/PAPER_READINESS_AND_REAL_WORLD_FUTURE_WORK.md prioritized missing controls
docs/RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md RRF and unseen-embedding timing
docs/EXPERIMENT_EXECUTION_STATUS.md exact B/C/E1 completion matrices
docs/CANDIDATE_HEADROOM_PROTOCOL.md read-only retrieval-headroom contract
docs/CANDIDATE_HEADROOM_RESULTS.md candidate ceilings and missing-gold reachability
docs/CROSS_PAPER_LEARNING_LEDGER.md Paper-1/Paper-2 findings and backport discipline
docs/TERMINOLOGY_AND_POSITIONING.md publication naming, overlap, and track fork
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
