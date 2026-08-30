# Paper-Readiness and Real-World Ranker Audit

Status: **living readiness document**. Nothing here changes the sealed
six-dataset confirmation. Package A1/A2/A3 were executed under new frozen
protocols and separate outputs; Package A is now closed, and every remaining
experiment still requires its own preregistration.

Publication prose uses **QLS-MLP (Query-Local Structure MLP)**. The frozen
implementation key `sa_mlp` and existing SA-named artifacts remain unchanged;
see [the terminology and positioning note](TERMINOLOGY_AND_POSITIONING.md).

## Scope boundary

This paper studies post-retrieval ranking and the value of learned graph
message passing. It does **not** need to become a new retrieval engine.

The production-facing input contract is:

```text
query_id
query_embedding
dense_ranked_candidate_ids
splade_ranked_candidate_ids
optional upstream rank/score metadata
```

The standalone method owns candidate fusion, candidate graph induction, fixed
structural computation or learned message passing, scoring, and top-K output.
Raw query text, tokenization, query encoding, ANN search, and SPLADE index
serving are upstream concerns and remain outside the latency claim.

An **unseen real-world query** in this project therefore means a query embedding
and candidate rankings that were not used to train the ranker or build a
query-specific topology/feature cache. It does not require the standalone repo
to accept raw text.

## Executive audit

The current evidence is strong enough to establish a controlled six-dataset
result. The following gaps remain before a broad NeurIPS-style or real-world
claim is defensible.

| Priority | Gap | Why it matters | Required resolution |
| --- | --- | --- | --- |
| P2 | Fresh untouched confirmation | The same benchmark families informed sequential screens, model decisions, and fairness confirmation | Open only after Packages A–E and their hypotheses are frozen |
| P0 | Uncached unseen-embedding latency | Current speed ratios use packed per-query topology/features | Rebuild fusion, induced graph, and QLS summaries on demand for never-cached query IDs |
| P0 | Edge provenance | `graph.pt` flattens native/title/KB and embedding-kNN edges | Re-export typed provenance and compare native-only, kNN-only, and union graphs |
| Complete | Strong trivial controls (A1/A2/A3) | QLS gains may be explainable by parameter-free rank/diffusion combinations | Rank, fixed-structure, and 19-parameter linear controls are complete; QLS retains a material lead on four datasets |
| P0 | Candidate-pool dependence | Results are conditional on a top-200+top-200 union and dataset-specific ceilings | Sweep frozen candidate budgets and report ceiling/effectiveness/latency together |
| P1 | Phase boundary and prediction | Six clean datasets show a boundary but do not yet explain or predict it | Run controlled topology/feature axes, freeze crossovers, and validate a predictor on held-out regimes |
| P1 | Upstream-quality robustness | QLS and GNN both depend on seed/candidate quality | Vary retriever agreement, seed corruption, embedding quality, and candidate recall |
| P1 | Inductive and OOD validity | Current queries share a fixed transductive corpus graph | Test unseen query distributions, entity/topic splits, and at least one external graph |
| P1 | Real-query failure modes | Current tasks underrepresent no-answer, ambiguity, and malformed/long-tail queries | Add frozen external embedding streams and confidence/abstention analysis |
| P1 | Relevance quality | Binary gold sets can be incomplete and some sources are effectively gold-heavy | Add candidate-conditional metrics, graded qrels where available, and a manual error audit |
| P1 | Full serving cost | Incremental GPU memory alone omits loaded graph/node stores and concurrency | Report full RSS/VRAM, p50/p95/p99, throughput, storage, and update costs |
| P2 | Downstream utility | Retrieval parity may not imply answer parity | Optionally evaluate a fixed reader at an identical context budget |
| P2 | Security and temporal drift | Hubs, injected edges, stale nodes, and new entities occur in deployed graphs | Add graph-poisoning and update/invalidation stress tests if space permits |

The six packages below are submission-critical, but they intentionally occur at
different priorities. P0 establishes controls, provenance, context budgets, and
honest online cost. P1 establishes mechanism and prediction. P2 freezes the
complete hypothesis/protocol and then opens the untouched external setting.

Current readiness is asymmetric: quality and clarity are strong, significance
is promising, and experimental rigor is strong for the sealed comparison.
Originality is the largest risk after the existing SA-MLP and contemporaneous
RTA work; systems evidence is incomplete until uncached timing is measured;
generalization is incomplete until an untouched retrieval-plus-graph setting is
evaluated; and mechanism is incomplete until the crossover is characterized.

## Canonical six-package plan

| Package | Priority | Purpose |
|---|---|---|
| A — semantic/structural decomposition | P0 | Determine which semantic, retrieval-prior, structural, and learned components recover the GNN benefit |
| B — edge provenance | P0, mandatory | Separate native/title/KB relations from embedding-kNN topology |
| C — structural-context budget | P0 | Measure quality and cost saturation at 50/100/200/400 candidates |
| D — online systems evaluation | P0 | Separate cached operator cost from uncached post-retrieval serving cost |
| E — robustness/crossover/predictor | P1 | Explain and predict `GNN effectiveness - QLS effectiveness` |
| F — untouched external confirmation | P2, last | Confirm frozen hypotheses and predictor on a new retrieval-plus-graph setting |

External confirmation is deliberately last. Opening it before A–E are fixed
would turn it into another development benchmark and weaken the confirmatory
claim.

## 1. Package F — fresh untouched confirmation (executed last)

Each frozen protocol evaluated its test split once, but the project as a whole
progressed through multiple sequential screens and confirmations on the same
benchmark families. The final QLS feature package and fairness controls were
developed with knowledge of earlier test outcomes.

This is not evidence of label leakage, and the repository records the sequence
transparently. It is nevertheless an adaptive-analysis risk. Additional seeds
on the same test queries do not create a fresh holdout.

Only after Packages A–E are complete and frozen:

1. freeze the final QLS-MLP, GNN family-selection rule, RRF rule, metrics, and
   systems protocol;
2. choose one dataset/query split whose test outcomes have never been inspected
   during method design;
3. generate its embedding and candidate artifacts upstream;
4. build a label-free graph without using relevance judgments;
5. register hashes before training or evaluation; and
6. evaluate all declared methods and seeds once.

The existing six datasets remain valuable development and broad replication
evidence. The fresh holdout provides the confirmatory endpoint.

## 2. Package D — online systems evaluation

### Main serving condition

For every held-out query, provide only:

```text
(query_embedding, dense_ranked_ids, splade_ranked_ids)
```

Then execute, without query-specific caches:

1. stable union or preregistered RRF;
2. candidate embedding gathering;
3. candidate-induced subgraph extraction;
4. shared seed construction;
5. QLS distance/path/connectivity/PPR computation, when applicable;
6. GNN or MLP forward pass; and
7. top-K selection.

Both methods may reuse corpus-static node embeddings, global CSR adjacency,
global static node statistics, and trained weights. Neither may read topology
or features keyed by the held-out query or its candidate set.

### Required query groups

Report effectiveness and latency for:

- ordinary in-distribution held-out embeddings;
- low Dense/SPLADE-overlap queries;
- low candidate-ceiling queries;
- short versus long candidate lists;
- low- versus high-degree candidate subgraphs;
- single- versus multi-answer queries;
- query-hop groups where trustworthy hop labels exist; and
- queries with no candidate-pool gold.

The last group measures behavior under upstream failure, not normal recall. It
should include confidence/abstention or an explicit “no in-pool answer” error
analysis rather than pretending a reranker can recover a missing gold.

### External real-query sources

External datasets can supply fresh queries and relevance labels, but they are
**not automatically valid external graph-retrieval settings**. The standalone
ranker still never needs raw text, yet a complete setting must contain all four
of: unseen query embeddings, frozen candidate rankings, relevance labels, and
either native topology or a label-free graph-construction rule frozen before
external test inspection. Possible query sources include:

- [Natural Questions](https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/),
  which originated from anonymized Google search queries and includes null
  answers;
- [MS MARCO](https://arxiv.org/abs/1611.09268), which originated from
  anonymized Bing queries and includes passage ranking and no-answer cases;
- [TREC Deep Learning](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf),
  for smaller but more deeply judged retrieval test sets;
- [BEIR](https://openreview.net/forum?id=wCu6T5xFjeJ), for heterogeneous OOD
  retrieval rather than a single QA distribution; and
- [QUEST](https://research.google/pubs/quest-a-retrieval-dataset-of-entity-seeking-queries-with-implicit-set-operations/),
  for natural multi-entity queries whose set-valued answers align with the
  coverage question.

Do not add all of them. Select one setting for which the missing graph contract
can be satisfied through native relations or a preregistered label-free rule
such as Wikipedia hyperlinks/title mentions, citation edges, or KB triples.
The minimum useful design is one untouched retrieval-plus-graph holdout; a
query dataset alone does not close the generalization gap.

## 3. Package B — edge provenance and the embedding-derived topology issue

The current `graph.pt` adjacency merges multiple sources and discards their
types. It may contain:

- title-mention or document-structure edges;
- native MetaQA/WebQSP KB edges; and
- kNN edges constructed from the same GTE-Qwen node embeddings supplied as
  model features.

The last case is scientifically important. A kNN graph derived from the input
embedding is not independent relational information; it is a discrete
transformation of the same semantic representation. The current result remains
valid for the frozen mixed graph, but it cannot establish whether native graph
structure or embedding-derived neighborhoods caused the effect.

Re-export, without editing CRAG, a standalone edge-provenance contract:

```text
edge_index
edge_source in {native_or_title, embedding_knn}
edge_type where available
direction
source_hash
```

Then compare:

1. native/title edges only;
2. embedding-kNN edges only;
3. their union;
4. union with type indicators;
5. kNN built with the feature encoder;
6. kNN built with a different frozen encoder.

This decomposition is required before making an edge-semantics claim. It also
creates the cleanest test of whether message passing contributes relational
information beyond smoothing in embedding space.

## 4. Package A — semantic versus structural decomposition

**Execution update (2026-08-30):** A1, A2, and A3 are complete. A1 reports Dense,
SPLADE, equal RRF, and validation-selected weighted RRF in
`P0_RANK_CONTROLS_RESULTS.md`. A2 reports frozen distance, PPR,
path/connectivity, structural-summary, and locked RRF+structure rules in
`P0_FIXED_STRUCTURAL_CONTROLS_RESULTS.md`. A3 reports the frozen 19-parameter
linear rank+structure scorer in `P0_LINEAR_RANK_STRUCTURE_RESULTS.md`. Package
A is closed to further tuning on these tests.

The final table should include a compact ladder that shows where the gain first
appears:

```text
Dense rank
SPLADE rank
equal RRF
query/node cosine over the union
seed distance or PPR alone
fixed RRF + PPR/distance fusion
linear scorer over frozen rank + structural features
seed-only MLP
QLS-MLP
seed-aware GNN
```

Reasons:

- RRF tests whether simple retrieval fusion explains the gain;
- PPR/distance alone tests whether learning is needed at all;
- the linear rank+structure control tests whether nonlinear/embedding capacity matters;
- RRF+PPR tests whether a parameter-free rank fusion matches QLS-MLP;
- the GNN comparison then isolates learned message passing rather than weak
  baselines.

All parameter-free rules and the learned linear control were frozen or
validation-selected from declared grids and were not tuned on test. That
boundary remains mandatory for later work.

## 5. Package C — structural-context budget and upstream dependence

The main result currently uses the union of Dense top-200 and SPLADE top-200.
That choice affects candidate ceiling, graph density, QLS computation, GNN
compute, and answer multiplicity.

Use preregistered post-fusion budgets 50, 100, 200, and 400 (or the full union
when fewer than 400 unique candidates exist).
At each budget report:

- candidate ceiling;
- mean and tail candidate count;
- induced nodes/edges, graph density, and connected components;
- R@1/R@5/R@20, MRR, FullCov, and nDCG where labels permit;
- cached and uncached post-retrieval latency; and
- full memory.

Also report QLS-specific fixed-summary compute and GNN-specific propagation
compute separately, in addition to shared fusion and graph-induction cost.

Do not select a budget separately for QLS-MLP and GNN. A budget is part of the
shared input contract.

Separately perturb upstream quality without changing labels:

- drop or replace a fixed fraction of top-ranked seeds;
- vary Dense/SPLADE overlap;
- use Dense-only, SPLADE-only, and RRF seeds;
- inject lower-ranked candidates while holding pool size fixed;
- degrade or replace the query embedding encoder; and
- stratify by candidate ceiling.

This determines whether fixed summaries are robust to realistic retriever
drift or merely exploit unusually strong seeds.

## 6. Inductive, OOD, and dynamic settings

The present setting is mainly **new query, fixed corpus graph**. State this
explicitly. It does not establish performance for new nodes or a changing
graph.

Use three distinct evaluations:

1. **Query-inductive:** unseen query embeddings over the same fixed graph. This
   is the primary deployment setting for the paper.
2. **Distribution-inductive:** entity/topic/domain-disjoint query embeddings
   with the same feature contract.
3. **Graph-inductive:** new nodes/edges or a held-out graph, requiring static
   feature refresh and possibly index updates.

OOD splits should distinguish covariate shift from concept/label shift where
possible, following the motivation of graph OOD benchmarks such as
[GOOD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html).

Dynamic update tests should measure:

- time to insert nodes and edges;
- which global/static values must be recomputed;
- cache invalidation volume;
- stale-feature effectiveness before refresh; and
- QLS versus GNN behavior under the same stale graph.

This is P1 systems evidence, not a prerequisite for the initial fixed-graph
claim.

## 7. Real-query failure modes

Real query embeddings can represent cases underrepresented by the current
benchmarks:

- no answer in the corpus;
- answer in the corpus but absent from the candidate pool;
- ambiguous or multi-intent information needs;
- entity aliases, misspellings, terse queries, and paraphrases;
- long-tail or new entities;
- multi-answer set retrieval; and
- conversational follow-ups whose embedding depends on previous turns.

The standalone project should receive upstream embeddings/candidates for these
cases rather than implementing text processing. Report:

- recall/coverage where reliable golds exist;
- calibrated probability that at least one relevant candidate is present;
- expected calibration error or Brier score;
- selective accuracy/coverage when abstaining;
- failure rate by query group; and
- qualitative error categories from a preregistered sample.

No-answer and missing-candidate cases must not be silently dropped from the
real-world analysis.

## 8. Relevance and evaluation quality

Some current sources are effectively gold-heavy, while others have low
candidate ceilings or incomplete judgments. A model can retrieve an unlabelled
but useful passage and be counted wrong.

For the fresh holdout:

- prefer graded or deeply judged qrels when available;
- report binary Recall/MRR and graded nDCG separately;
- report candidate-conditional and all-gold metrics together;
- audit duplicate/near-duplicate passages;
- manually adjudicate a frozen stratified sample of QLS/GNN disagreements; and
- keep human/LLM-assisted judgments blinded to model identity.

Automatic LLM judgments may assist error categorization but should not be the
sole headline relevance source.

## 9. Systems completeness within the ranker boundary

The systems table should report more than incremental GPU allocation:

- batch-1 and batch-16 p50/p95/p99;
- fixed-concurrency throughput;
- full process RSS and total VRAM;
- global graph, node-embedding, model, and feature storage;
- CPU thread count and GPU utilization;
- on-demand graph-induction and QLS-feature time separately;
- latency by candidate count and edge count;
- cold model/data-store load time separately; and
- optional energy/query and cloud cost per million ranked queries.

Also compute the cache break-even point:

\[
N_{break-even} =
\frac{\text{precompute or refresh cost}}
{\text{on-demand cost saved per repeated query}}.
\]

For mostly unique real-world queries, per-query feature caches may provide
little value. Corpus-static values remain reusable.

## 10. Package E — robustness, crossover, and utility prediction

This is the main scientific extension for a NeurIPS main-track submission.
Controlled interventions should weaken seed quality, remove high-ranked seeds,
increase Dense/SPLADE disagreement, inject irrelevant candidates, corrupt or
rewire graph edges, remove native structural edges, vary kNN density, and weaken
semantic embeddings.

For every regime record QLS and GNN changes separately and the primary target:

```text
message-passing utility = GNN effectiveness - QLS effectiveness
```

Candidate explanatory variables include:

- candidate ceiling and seed recall;
- Dense/SPLADE disagreement;
- mean seed-to-candidate distance and number of connected seeds;
- path redundancy and PPR concentration/entropy;
- candidate graph density, local clustering, and degree/hub exposure;
- semantic-neighborhood coherence;
- native-versus-kNN edge proportion; and
- graph/semantic agreement.

The predictor should answer whether paying for message passing is worthwhile
from properties available for the query, retrieved candidates, and induced
graph. Do not choose its architecture yet. Train and tune only on development
settings, then evaluate on held-out regimes or datasets.

### Mechanistic and future-theory question

> Under what conditions are fixed query-local graph summaries sufficient for
> preserving the ranking information that a message-passing model obtains from
> the candidate graph?

The working **structural-compressibility hypothesis** is that seed membership,
shortest distance, reachable-seed count, path multiplicity, diffusion/PPR, and
connectivity may suffice when relevance primarily depends on those statistics.
The complementary **rich-content hypothesis** predicts a GNN advantage when
relevance depends on rich neighbor semantics, ordered/compositional or typed
relations, or higher-order content transformations. These are future
mechanistic hypotheses, not theorems.

## 11. Downstream utility: optional, not core

This paper can remain a retrieval paper. If a downstream check is added, keep
it controlled:

- one frozen reader or generator;
- identical top-K and token budget;
- no model-specific prompting;
- answer EM/F1 or task-specific correctness;
- citation/support coverage; and
- retrieval plus reader latency reported separately.

This test asks whether small retrieval differences matter to answer quality. It
must not become a second C-RAG systems contribution.

## 12. Recommended stopping order

To avoid scope expansion, resume in this order:

### P0.1 — semantic/structural controls

Implement equal RRF and parameter-free structural baselines on frozen
validation data, freeze their rules, and evaluate once.

### P0.2 — graph provenance

Re-export native/title versus kNN edge provenance and run native-only,
kNN-only, and union anchors for QLS and GNN.

### P0.3 — candidate/context budget

Run the shared 50/100/200/400 sweep and report candidate ceiling,
R@1/R@5/R@20, MRR, FullCov, induced graph size/density, and both method-specific
compute paths.

### P0.4 — uncached serving path

Implement on-demand graph induction and QLS computation. Require numerical
parity with the cached tensors, then measure unseen-embedding latency.

### P1 — robustness, phase diagram, and utility predictor

Run retrieval-seed, candidate, topology, and semantic-feature perturbations.
Fit the predictor on development settings and test it on held-out regimes or
datasets.

### P2.1 — freeze final hypotheses and protocol

Freeze Packages A–E, all target metrics, predictor inputs, graph-construction
rule, and external evaluation contract before accessing the external outcome.

### P2.2 — fresh untouched confirmation

Evaluate once on the untouched external retrieval-plus-graph substrate. This is
the final confirmatory gate, not an exploratory dataset.

Optional downstream QA, deeper formal theory, and additional GNN families must
not delay these steps.

## 13. Claims enabled by each gate

| Completed evidence | Defensible claim |
| --- | --- |
| Current confirmation only | Fixed structural summaries recover nearly all GNN R@5 under frozen candidates and warm per-query caches |
| + Package A controls | Whether semantic rank fusion, structural rules, linear scoring, or nonlinear QLS capacity is needed |
| + Package B provenance | Whether native relations, embedding kNN, or their union drive the result |
| + Package C context budget | How quality and cost saturate as structural context grows |
| + Package D uncached serving | Same comparison with real post-retrieval unseen-embedding cost |
| + Package E phase diagram/predictor | Predictive statement about when message passing is worth its cost |
| + Package F fresh holdout | Confirmatory generalization beyond adaptively studied test sets |
| + downstream check | Evidence that retrieval behavior transfers to answer utility |

The NeurIPS-level story should be built from Packages A–F, not from an attempt
to claim that MLPs universally beat GNNs.

Target claim—not established by the current checkpoint:

> We characterize measurable retrieval and graph regimes in which fixed query-
> local structural computation is sufficient and regimes in which learned
> neighborhood aggregation provides additional value.

For the NeurIPS **main track**, the minimum package is mechanism/phase diagram,
a predictor or strong explanatory framework, edge-provenance controls, complete
online systems evaluation, and untouched external confirmation. For the
**Evaluations & Datasets** track, a release-quality MPR-Bench can instead center
the four-level fairness decomposition, cross-dataset benchmark, retrieval-prior
confounding, graph provenance, systems protocol, and standardized message-
passing utility evaluation. Do not select the final track yet.
