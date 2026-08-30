# P0 A2 fixed structural-control protocol

Status: **frozen before A2 test evaluation**.

## Question

How much of QLS-MLP's gain over semantic rank fusion is recoverable from an
individual fixed query-local graph statistic, or from a locked combination of
semantic and structural rankings, with no learned parameter and no message
passing?

This is the second stage of Package A. A1 established Dense, SPLADE, equal RRF,
and validation-selected weighted RRF on the unchanged candidate union. A2 uses
the already sealed QLS feature cache to isolate simple graph signals. It does
not change QLS-MLP or any GNN.

## Frozen methods

Every method ranks the full unchanged candidate union. Ties are resolved by
ascending global node ID.

- **Selected RRF:** the validation-selected Dense/SPLADE weight already fixed
  independently in A1.
- **Distance:** fixed scores 1, 2/3, 1/3, and 0 for distance buckets 0, 1, 2,
  and 3-or-unreachable.
- **PPR:** the exact eight-iteration retrieval-seed-conditioned PPR scalar in
  the frozen QLS cache.
- **Path/connectivity:** the arithmetic mean of seed connections, directed
  path counts of length 1/2/3, and common-out-neighbor connectivity.
- **Structural summary:** the arithmetic mean of distance score, the five
  path/connectivity components, and PPR.
- **Selected RRF + PPR:** locked equal reciprocal-rank fusion of the two full
  rankings, with constant 60.
- **Selected RRF + structural summary:** the same locked fusion using the full
  structural-summary ranking.

All structural inputs are inference-safe functions of frozen retrieval seeds
and graph topology. Gold/support labels, hop labels, question nodes, and target
test outcomes are prohibited from feature construction and scoring.

## Alignment and leakage gates

Before metrics are accepted, the evaluator must demonstrate:

1. exact structural feature names and cache format;
2. equality of the cache source fingerprint and structural contract with the
   sealed fairness confirmation;
3. exact candidate-row counts;
4. equality of a SHA-256 digest covering every query ID and candidate ID in
   stable local order; and
5. reproduction of A1's selected-RRF test metrics.

The evaluator loads neither graph edges nor embeddings. It reads the frozen
rank arrays, labels/splits for evaluation, and the existing memory-mapped QLS
feature cache. The original CRAG repository remains read-only.

## Statistical and selection boundary

There are no trainable parameters, seeds, hyperparameters, or A2 validation
selection. Every fixed rule above is reported on every dataset. Any uncertainty
interval added later is descriptive unless separately preregistered; the main
A2 table is deterministic.

## Systems boundary

A2 runtime is offline artifact evaluation, not a service-latency result. The
existing structural cache is intentionally reused to avoid recomputation.
On-demand feature construction remains Package D and cannot be inferred from
this run.

## Stop condition

After all six datasets pass alignment and the complete table is compiled, A2
is closed. A learned linear QLS control is A3 and requires a new frozen
protocol before training or test evaluation. Edge provenance, candidate-budget
sweeps, perturbations, and architecture changes remain out of scope.
