# P0 Package A3: Linear Rank + Structure Protocol

Status: **frozen before any A3 test evaluation**. The machine-readable source
of truth is `configs/p0_linear_rank_structure.yaml`.

## Question

Package A1 showed that rank fusion helps but does not explain QLS-MLP. Package
A2 showed that fixed graph summaries contain real signal but that a single
training-free rule also does not explain QLS-MLP. A3 now asks the narrow next
question:

> Can learned *linear weighting* of the same frozen rank and graph summaries
> explain the result, without embeddings, hidden layers, adjacency in the
> learned forward pass, or message passing?

This is a capacity control, not a proposed architecture.

## Exact model

For candidate `c` and query `q`, A3 computes

`score(q,c) = w^T x(q,c)`.

The 19 inputs, in fixed order, are:

1. Dense reciprocal-rank feature;
2. SPLADE reciprocal-rank feature;
3. the seven sealed graph-wide static features; and
4. the ten sealed query-local structural features.

For one-based source rank `r`, a present candidate receives
`(60 + 1) / (60 + r)` and an absent candidate receives zero. Static and local
features are consumed exactly in the sealed cache metadata order. The linear
layer has **19 trainable weights and no bias**. All weights initialize to zero.
There are no embeddings, feature projections, hidden layers, nonlinearities,
or learned neighborhood operations.

A shared scalar bias is deliberately excluded: it cancels from a per-query
softmax and cannot affect within-query ranking, so counting it as usable model
capacity would be misleading.

## Immutable inputs and leakage boundary

A3 uses the exact A1/A2 candidate IDs and order, canonical train/validation/test
splits, gold sets, and sealed QLS structural cache. The graph and embedding
matrices are not loaded by the A3 evaluator. Its optional derived cache stores
only global candidate IDs and the two label-free source-rank features; it never
stores gold labels.

Gold membership is constructed in memory only for the current run. Training
golds enter only the training loss, validation golds enter only learning-rate
and epoch-checkpoint selection, and test golds enter only the single final
evaluation of each canonical seed.

## Frozen optimization

- Seeds: 0, 1, 2, 3, 4.
- Epochs: 3.
- Batch size: 512 queries.
- Loss: the same multi-positive listwise objective used by the sealed models,
  `logsumexp(all candidate scores) - mean(positive candidate scores)`.
- Optimizer: AdamW, zero weight decay, gradient norm clipped to 1.0.
- Learning-rate grid: `0.001`, `0.01`, `0.05`.
- Learning-rate selection: seed 0, validation R@5 only; ties choose the smaller
  rate. The selected seed-0 checkpoint is reused rather than retrained.
- Epoch checkpoint: validation R@5 only; ties choose the earlier epoch.
- Ranking ties: ascending global node ID.
- Test: exactly once per seed after validation selection; all six datasets and
  all registered metrics must be reported.

## Interpretation gates

The result cannot support “MLP beats GNN everywhere.” It distinguishes three
mechanisms:

- If A3 matches QLS-MLP, the useful information is largely present in fixed
  ranks and graph summaries, and nonlinear semantic interaction is unnecessary.
- If A3 beats A1/A2 but trails QLS-MLP, learned feature weighting matters, while
  the remaining gap is attributable to nonlinear capacity and/or embedding
  interaction—not automatically to message passing.
- If this 19-parameter control matches the seed-aware GNN, learned message
  passing is unnecessary in that candidate-ranking regime.

The complete ladder remains: rank-only, fixed structure, learned linear fixed
structure, seed-only MLP, QLS-MLP, and seed-aware selected GNN. No architecture,
feature, perturbation, or dataset-dependent rule may be added inside A3.

## Systems accounting

Warm-cache inference latency, throughput, GPU memory, training time, derived
cache build time, and derived cache bytes are reported separately. These
numbers do not replace the still-deferred uncached unseen-embedding systems
benchmark.
