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


## Required headroom companion

Every budget cell must be reported beside its candidate ceiling from
`CANDIDATE_HEADROOM_PROTOCOL.md`. The ceiling is `min(p, K) / g`, not the pool
coverage `p / g`.

The ceiling **rises monotonically with budget on all six datasets**, so a raw
metric gain from budget 50 to 400 is partly the ceiling moving rather than the
model reranking better. On MetaQA the Recall@5 ceiling grows from 0.2104 at
budget 50 to 0.3262 at budget 400; no model choice may be credited with that
0.116. Budget effects are therefore reported as attainment against the
per-budget ceiling as well as in absolute terms.

The paired `GNN - QLS` contrast is unaffected: both models receive identical
pools at every budget, so the ceiling cancels in the paired difference. Only the
absolute levels and any cross-dataset comparison need the ceiling.

This is a reporting requirement. It does not change the frozen budget contract,
the fusion rule, or any candidate pool, and no budget may be selected using it.

### Implementing script

`scripts/compile_package_c.py` joins a finished Package C analysis with the
headroom diagnostic and emits the combined report to
`docs/CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md`. It is read-only: it consumes
finished outputs, trains nothing, and modifies no candidate pool. It refuses a
Package C analysis that is not complete and a headroom file whose contract does
not declare `candidate_pools_modified: false`.

It reports coverage, GoldFraction, AnyGold, AllGold, the per-budget ceilings,
both models' metrics, and `ceiling - metric` with attainment beside them.

For the requirement that budget effects be separated from ceiling movement, it
decomposes each budget step exactly. Recall is `attainment x ceiling`, so

```text
d_recall = a0 * (c1 - c0)  +  c1 * (a1 - a0)
           ceiling effect      ranking effect
```

The two terms sum to the observed change with no residual. The split is
order-dependent by construction, so the ordering is fixed in code before any
Package C result was opened and cannot be chosen afterwards to suit an
attribution. Where a pool contains no reachable gold the ceiling is zero, and
the step is reported as unattributable rather than as a zero effect.
