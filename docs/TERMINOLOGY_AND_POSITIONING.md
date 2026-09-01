# Terminology, novelty boundary, and submission positioning

## Working identity

**Working paper title:** *When Does Graph-Aware Retrieval Need Message Passing?*

**Working subtitle:** *Fixed Structural Summaries versus Learned Neighborhood Aggregation*

**Target one-line identity:** We characterize when graph-aware retrieval benefits from
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

The primary question is narrower than “MLPs beat GNNs”:

> **Once retrieval has already happened, how much learned message passing is
> still necessary to rank graph-aware evidence?**

Equivalently: given an upstream retriever, frozen candidates, and a graph, what
additional benefit comes from learned neighborhood aggregation beyond the
retrieval prior and explicit query-local structure?

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

| Dimension | RTA | This project |
|---|---|---|
| Task | Node prediction/text-attributed graph learning | Query-candidate retrieval ranking |
| Starting object | Graph node | External query |
| Context retrieval | Builds its own retrieval context | Upstream Dense/SPLADE already produced candidates |
| Structural signal | Node-conditioned PPR | Retrieval-seed-conditioned distance/path/PPR |
| Labels as inference context | Possible | Prohibited |
| Neighbor representation aggregation | Yes | QLS consumes fixed summaries |
| Main question | Can retrieval replace message passing? | What does message passing add after retrieval? |

The query conditioning differs:

```text
RTA: PPR(node u -> graph)
QLS: PPR(retrieval seeds S(q) -> candidate d)
```

This project does not claim to have invented PPR-based structural retrieval.

The novelty therefore rests on the retrieval-specific decomposition, controlled
phase diagram, edge-provenance audit, and cost accounting—not on a generic
“MLP replaces GNN” statement.

## Central mechanistic hypotheses

> **H1 — Structural compressibility:** QLS-MLP should approach or match learned
> message passing when graph information useful for ranking a candidate can be
> compressed into seed membership, seed distance, path multiplicity or
> connectivity, and seed-conditioned diffusion.

> **H2 — Rich-content requirement:** GNNs should gain an advantage when ranking
> depends on richer neighbor content, interactions, typed/compositional
> relations, or higher-order information that fixed query-local summaries do
> not represent adequately.

The intended result is not “QLS always beats GNN.” It is a prediction of when
fixed structural computation is sufficient and when message passing is worth
its additional cost. These hypotheses are not established theorems.

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

Target claim—not yet established:

> We characterize measurable retrieval and graph regimes in which fixed query-
> local structural computation is sufficient and regimes in which learned
> neighborhood aggregation provides additional value.

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

Status labels used here: **COMPLETE** (finished, integrity-verified, frozen),
**RUNNING** (registered cells executing now), **GATED** (frozen but blocked on a
named dependency), **FUTURE** (not preregistered). Exact per-cell counts live in
[`EXPERIMENT_EXECUTION_STATUS.md`](EXPERIMENT_EXECUTION_STATUS.md) and are not
duplicated here.

Do not expand model design until these are complete:

1. **Package A — semantic versus structural decomposition. COMPLETE.** Dense,
   SPLADE, equal RRF, validation-only weighted RRF, structural-only
   PPR/distance/path, RRF+structural combinations, linear QLS, QLS-MLP, and
   seed-aware GNN.
2. **Package B — edge provenance (mandatory). RUNNING.** Native/title/KB-only,
   embedding-kNN-only, and union graphs for both QLS and GNN, with the union
   proven equivalent to the frozen adjacency.
3. **Package C — structural-context budget. RUNNING.** Shared 50/100/200/400
   candidate budgets with ceiling, effectiveness, graph size/density,
   method-specific compute, and total post-retrieval latency.
4. **Package D — online systems evaluation. GATED on all six Package C
   budget-400 cells.** Separate cached operator latency from an uncached path
   beginning at unseen query embeddings and Dense/SPLADE rankings. Report batch
   1/16, p50/p95/p99, throughput, GPU/CPU memory, storage, cold start, and
   cache break-even.
5. **Package E — robustness phase diagram and utility predictor. E1 RUNNING; E2
   GATED on a complete E1 screen, a validation-only rate selection, a commit,
   and a new freeze tag.** Perturb seed quality, retriever agreement, candidate
   noise, topology, kNN density, and semantic features; predict `GNN
   effectiveness - QLS effectiveness` on held-out regimes or datasets.
6. **Package F — fresh untouched confirmation. SEALED / UNOPENED.** Only after
   A–E and their hypotheses are frozen, evaluate an external setting with
   unseen query embeddings, upstream candidates, relevance labels, and native
   or preregistered label-free topology.

The priority sequence is:

```text
P0: A controls, B provenance, C context budget, D online systems
P1: E robustness, phase diagram, utility predictor
P2: freeze hypotheses/protocol, then F untouched confirmation
Optional: downstream QA, deeper formal theory, additional GNN families
```

## Claim restrictions

Until the packages above are complete, do not claim that:

- QLS-MLP or an MLP universally beats GNNs;
- message passing is unnecessary in general;
- the model is topology-free;
- the existing 2.49--7.08x number is end-to-end or uncached latency;
- external query datasets alone establish graph-retrieval generalization;
- fewer parameters explain the result—the principal frozen models are
  approximately parameter matched;
- a dataset with low absolute recall is therefore a dataset where message
  passing or QLS failed—MetaQA attains roughly 92% of its candidate ceiling, so
  its low raw Recall@5 is dominated by upstream candidate generation, not by
  reranking;
- candidate coverage `p / g` is an oracle Recall@K—the achievable value is
  `min(p, K) / g`, and the two differ whenever a query has more golds than the
  cut-off;
- absolute metric levels are comparable across datasets—they are confounded by
  candidate coverage unless each is reported beside its ceiling.

The dataset that currently retains the most genuine reranking headroom is
WebQSP, at roughly 74% ceiling utilization, not the datasets with the lowest raw
scores. See [`CANDIDATE_HEADROOM_RESULTS.md`](CANDIDATE_HEADROOM_RESULTS.md).

The current systems result is specifically a **warm-cache operator/reranking
comparison**. It remains valid, but it answers a smaller question than an
uncached post-retrieval deployment benchmark.
