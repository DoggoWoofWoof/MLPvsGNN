# Package C: Candidate and Structural-Context Budget

Status: **frozen before any Package C test result is opened**.

## Question

How do candidate ceiling, induced graph context, effectiveness, and compute
change when QLS-MLP and the seed-aware selected GNN receive the same increasing
candidate budget?

The clean result uses the stable union of Dense top-200 followed by unseen
SPLADE top-200 candidates. Directly truncating that order would make budgets
50/100/200 effectively Dense-only. Package C therefore uses the already frozen
parameter-free equal-RRF rule as a common post-fusion order:

\[
s(d)=\frac{\mathbf{1}[d\in D]}{60+r_D(d)}+
\frac{\mathbf{1}[d\in S]}{60+r_S(d)}.
\]

Ties use ascending global node ID. The common budgets are 50, 100, 200, and
400; if the union has fewer unique candidates, the full union is used. No
budget is chosen per model, and no test metric may select a budget.

## Frozen seed and graph contract

The seed prior remains the original union of Dense top-5 and SPLADE top-5,
intersected with the budgeted pool. This isolates candidate/context size rather
than introducing an RRF-seed redesign. Both models receive the same retained
seed identities.

The global graph is the exact sealed A multigraph used by the clean
confirmation. Package B studies edge provenance separately; Package C changes
only the equal-RRF candidate budget.

For every dataset and budget, QLS features and candidate-induced adjacency are
rebuilt from scratch under a budget-specific hash. QLS-MLP and the frozen
validation-selected seed-aware GNN are retrained at seeds 0/1/2/3/4 with the
same embeddings, labels, splits, loss, optimizer, epoch count, and parameter
regime.

## Required outputs

Each budget reports:

- equal-RRF effectiveness and candidate ceiling;
- QLS and GNN R@1/R@5/R@20, MRR, and FullCov@20;
- candidate count mean/p50/p95/p99/max;
- stored directed induced-edge count and density at the same quantiles;
- connected-component count at the same quantiles;
- topology construction time and storage;
- QLS static/local feature construction time and storage;
- cached inference latency, training time, GPU memory, and CPU RSS; and
- model checkpoints needed for the separately timed uncached Package D path.

Candidate ceiling is a label-dependent oracle diagnostic and cannot be an
input to a deployable method-selection predictor.

## Interpretation

The primary curve is paired `GNN - QLS` R@5 versus budget, plotted alongside
candidate ceiling, induced edges, and latency. A crossover would show that the
relative value of learned propagation depends on how much structural context
is admitted. Saturation without crossover is also informative: it identifies
the cheapest common budget that retains effectiveness, but the paper will not
retroactively relabel that point as the sole primary result.

