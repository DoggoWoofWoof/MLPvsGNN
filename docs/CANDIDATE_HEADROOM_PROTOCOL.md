# Candidate-Generation Headroom Diagnostic

Status: **companion reporting layer, frozen before any headroom number is
opened**. This is not a change to a frozen protocol. It adds no experimental
condition, trains no model, and touches no result that Packages A-F report.

## Question

A ranking metric answers "how well were the candidates ordered". It cannot say
whether the gold evidence was ever inside the candidate universe. Every reported
number in this paper is therefore ambiguous between three failures:

1. **Upstream candidate generation** - the gold node is absent from the pool, so
   no reranker could ever retrieve it.
2. **The reporting cut-off K** - the gold node is present, but the query has
   more golds than K positions.
3. **Reranking** - the gold node is present and reportable, and the model still
   ranked it below the cut-off.

Only (3) is what this paper studies. QLS-MLP and the matched seed-aware GNN must
not be credited or penalised for (1) or (2).

## The ceiling is not the coverage

For one query with `g` gold nodes, `p` of them present in the pool, and cut-off
`K`, the largest achievable Recall@K is

    max Recall@K = min(p, K) / g

Two distinct caps are folded into that number. Reporting pool coverage `p / g`
as though it were an oracle Recall@K overstates the achievable value whenever
`g > K`. The diagnostic therefore always reports the caps apart:

| Field | Meaning |
| --- | --- |
| `recall_ceiling@K` | `mean(min(p, K) / g)` - both caps |
| `recall_ceiling_perfect_retrieval@K` | `mean(min(g, K) / g)` - cut-off cap only |
| `recall_headroom_lost_to_candidate_generation@K` | the gap between them |
| `gold_fraction_at_pool_macro` | `mean(p / g)` - pool coverage, *not* an oracle R@K |

`hit_ceiling@K` and `ndcg_ceiling@K` are reported on the same basis.

## Pools compared

Each dataset and split reports the same headroom block for:

- `dense_top200` and `splade_top200` alone, plus a `source_complementarity`
  block giving what the union buys over the better single source;
- `frozen_union` - the exact pool the clean results were computed on;
- `equal_rrf_budget_{50,100,200,400}` - built by the same equal-RRF rule as
  Package C (`rrf_constant` 60, weights 0.5/0.5, ties by ascending global node
  ID, full union when the unique count is below the budget). The budget pools
  are reconstructed by mirroring `candidate_budget.build_budget_dataset`; a
  unit test asserts the reconstruction is identical to it row for row.

## Missing-gold reachability

For every gold node absent from a frozen pool, a read-only bounded breadth-first
search starts at the frozen retrieval seeds (Dense top-5 union SPLADE top-5) and
walks the undirected view of the global graph to at most three hops. Missing
golds are placed in disjoint buckets 1 / 2 / 3 / beyond-3-or-unreachable, and
the cumulative reachable-within-`h` counts are reported beside them.

A per-query frontier budget bounds the walk. Queries that hit the budget are
counted in `frontier_capped_queries` and excluded from the resolved
denominators; they are never reported as unreachable, because that would
manufacture a finding out of a compute limit.

The stored graph orientation is not modified. `stored_graph_was_symmetric`
records whether symmetrising the view changed anything.

## What this diagnostic must never do

- It must not admit a reachable missing gold into a Paper-1 candidate pool.
- It must not expand, reorder, regenerate, or re-rank any pool.
- It must not change a frozen hash, config, protocol, model input, or reported
  primary result.
- It must not be used to select a budget, a model, a perturbation rate, or a
  dataset.
- Candidate expansion remains a separate Paper-2 / G2 research question. That a
  missing gold sits one hop from a seed is a *measurement*, not a licence to go
  and fetch it.

The runner re-validates the frozen candidate contract against the registered
confirmation before computing anything, and re-checks the contract hash after
it finishes. Any drift is an error, not a result.

## Interpretation rules

Report every primary metric beside its ceiling on the same split and cut-off.

- A model gap that survives **below a shared ceiling** is a reranking result and
  is what Q3 is about.
- A metric **pinned at its ceiling** is an upstream candidate-generation result
  and must be labelled as such. It is not evidence about message passing.
- A dataset whose ceiling is low is not a dataset where message passing failed.
  WebQSP in particular cannot be interpreted until its candidate ceiling is put
  beside its reported Recall@5.
- Two datasets are only comparable as reranking evidence once their ceilings are
  stated. Absolute metric levels across datasets are otherwise confounded by
  candidate generation.

## Execution

Registered entrypoint:

    python experiments.py run candidate-headroom --detach

CPU-only, so it is safe to run beside an in-flight GPU package. Output lands in
`outputs/candidate_headroom/<dataset>.json` locally and under
`outputs/candidate_headroom/<dataset>/<fingerprint16>/headroom.json` on the
Modal volume. The runner is idempotent: a complete diagnostic is reused, and a
record written under a different diagnostic contract is an error rather than
something to overwrite.
