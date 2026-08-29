# Terminology, novelty boundary, and submission positioning

## Working identity

**Working paper title:** *When Does Graph-Aware Retrieval Need Message Passing?*

**Working subtitle:** *Fixed Structural Summaries versus Learned Neighborhood Aggregation*

**One-line identity:** We characterize when graph-aware retrieval benefits from
learned neighborhood propagation and when fixed query-local structural
computation is sufficient.

The title is intentionally not frozen. A contemporaneous paper published in
August 2026 makes nearby claims about interpreting message passing as retrieval,
so the final title must be checked again before submission.

## Publication name and immutable artifact key

The publication-facing name of our fixed-structure model is:

> **QLS-MLP — Query-Local Structure MLP**

`SA-MLP` is already the exact name of a TMLR method, *SA-MLP: Distilling Graph
Knowledge from GNNs into Structure-Aware MLPs*. Reusing that name would create
an avoidable attribution and searchability problem.

For reproducibility, the historical implementation and sealed artifacts retain
their original identifiers:

| Publication-facing term | Immutable repository/artifact identifier |
|---|---|
| QLS-MLP | `sa_mlp` |
| QLS feature cache | `sa_features` and existing SA-named files |
| QLS confirmation protocol/results | `SA_MLP_CONFIRMATION_*` filenames |
| QLS frozen tags/configs | existing SA-named tags and configuration keys |

Do not mass-rename code, configurations, result files, tags, hashes, or sealed
protocols. Paper tables and new prose must say **QLS-MLP**; an artifact appendix
must state that `sa_mlp` is the frozen legacy key for the same model.

## Scientific question and four-level decomposition

The defensible question is narrower than “MLPs beat GNNs”:

> **For graph-aware candidate retrieval, what portion of the benefit attributed
> to GNN message passing comes from upstream retrieval priors and explicit
> query-local structural statistics?**

The mandatory decomposition is:

```text
plain MLP
  -> seed-only MLP
  -> QLS-MLP (fixed query-local graph computation)
  -> seed-aware selected GNN (learned message passing)
```

It identifies three different effects:

- `seed-only - plain` = value of the frozen upstream retrieval prior;
- `QLS-MLP - seed-only` = value of fixed query-local structural computation;
- `QLS-MLP - seed-aware GNN` = fixed summaries versus learned neighborhood
  propagation, with both models conditioned on the same seeds.

QLS-MLP is **non-message-passing**, not topology-free. It consumes graph-derived
distance, path-count, connectivity, and PPR summaries.

## Related-work boundary

The broad claim that an MLP can use graph context without learned message
passing is already crowded by SGC, SIGN, Graph-MLP, BUDDY, the existing SA-MLP,
and related decoupled/precomputed graph methods. The paper must not claim that
fixed graph computation or message-passing-free graph learning is itself new.

The closest new overlap is *Rethinking Message Passing as Retrieval for
Text-Attributed Graph Learning* (RTA, arXiv:2608.26732). Its setting and ours
must be distinguished explicitly:

| Axis | RTA | This project |
|---|---|---|
| Task | Text-attributed graph node/representation prediction | Query-to-candidate document/entity ranking |
| Upstream input | Label-aware contextual retrieval constructs a retrieval graph | Frozen Dense/SPLADE rankings and unseen query embeddings |
| Structural use | Retrieved-neighbor aggregation with an MLP | Explicit retrieval-seed-conditioned distance/path/PPR summaries |
| Empirical regimes | Text-attributed graph benchmarks | Six QA/KB retrieval regimes |
| Main control | Retrieval view of graph message passing | Four-level retriever-prior/structure/propagation decomposition |

The novelty therefore rests on the retrieval-specific decomposition, controlled
phase diagram, edge-provenance audit, and cost accounting—not on a generic
“MLP replaces GNN” statement.

Primary references:

- [RTA: Rethinking Message Passing as Retrieval for Text-Attributed Graph Learning](https://arxiv.org/abs/2608.26732)
- [SA-MLP: Distilling Graph Knowledge from GNNs into Structure-Aware MLPs](https://openreview.net/forum?id=MZ2kKZc8m7)
- [SGC](https://proceedings.mlr.press/v97/wu19e.html)
- [SIGN](https://arxiv.org/abs/2004.11198)
- [Graph-MLP](https://arxiv.org/abs/2106.04051)
- [BUDDY](https://arxiv.org/abs/2209.15486)

## Submission fork

### NeurIPS main track

The main-track version needs a strong explanatory result beyond the six-dataset
table: a reproducible phase diagram, a validated crossover predictor, or useful
theory showing when learned propagation helps, is neutral, or hurts. Originality
is currently the largest risk because the broad MLP-versus-message-passing area
has substantial prior work.

### NeurIPS Evaluations & Datasets track

An Evaluations & Datasets submission can make the protocol itself central. A
possible contribution is **MPR-Bench**, provided the artifacts meet the track's
release and documentation expectations:

> Existing graph-retrieval comparisons confound upstream retriever priors,
> access to graph information, and learned propagation. MPR-Bench separates
> these factors and shows that the scientific conclusion changes when the
> comparison is controlled.

This route still requires high-quality data provenance, reproducible evaluation,
strong controls, and an honest limitations analysis. It should not be treated as
an easier version of the main-track submission.

Planning references:

- [NeurIPS 2026 Evaluations & Datasets call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- [NeurIPS 2026 Evaluations & Datasets reviewer guidance](https://nips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines)

## Six submission-critical experiment packages

Do not expand model design until these are complete:

1. **Fresh untouched external retrieval-plus-graph setting.** It must provide
   queries, frozen candidates, labels, and either native topology or a
   label-free graph construction rule frozen before external test inspection.
   NQ, MS MARCO, and BEIR are query sources, not automatically complete graph
   retrieval settings.
2. **Uncached post-retrieval systems timing.** Begin with an unseen query
   embedding plus Dense and SPLADE ranked IDs. Include fusion, candidate-set
   construction, graph induction, fixed summaries or learned propagation, model
   scoring, and top-K. Report batch 1/16, p50/p95/p99, throughput, GPU memory,
   CPU memory, storage, preprocessing, and cold start. Raw query encoding is out
   of scope.
3. **Strong non-neural controls.** Dense, SPLADE, equal-weight RRF,
   validation-only weighted RRF, distance/PPR, RRF+PPR, and a linear scorer over
   the fixed structural features.
4. **Edge-provenance intervention.** Re-export edge sources and compare
   native/title/KB-only, embedding-kNN-only, and their union for both QLS-MLP
   and the seed-aware GNN.
5. **Candidate-budget sweep.** Freeze budgets such as 50/100/200/400 and report
   candidate ceiling, R@5, R@20, induced nodes/edges, QLS computation, and GNN
   computation. The same budget must be used for both methods.
6. **Deeper phase diagram and crossover predictor.** Vary topology quality,
   feature quality, hubness, graph dependence, answer multiplicity, and degree;
   preregister the final levels and evaluate prediction on held-out regimes.

## Claim restrictions

Until the packages above are complete, do not claim that:

- QLS-MLP or an MLP universally beats GNNs;
- message passing is unnecessary in general;
- the model is topology-free;
- the existing 2.49--7.08x number is end-to-end or uncached latency;
- external query datasets alone establish graph-retrieval generalization;
- fewer parameters explain the result—the principal frozen models are
  approximately parameter matched.

The current systems result is specifically a **warm-cache operator/reranking
comparison**. It remains valid, but it answers a smaller question than an
uncached post-retrieval deployment benchmark.
