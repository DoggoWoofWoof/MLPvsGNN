# Package A1 protocol: frozen rank-only controls

Status: **frozen before test evaluation**.

This protocol starts the resumed P0 program without changing the sealed
six-dataset QLS-MLP confirmation. It evaluates no model checkpoint and loads no
graph. The original CRAG repository remains a read-only source of frozen ranked
candidate IDs and query manifests.

The machine-readable contract is
[`configs/p0_rank_controls.yaml`](../configs/p0_rank_controls.yaml).

## Question

On the unchanged Dense-top-200 union SPLADE-top-200 candidate contract, how much
effectiveness is already obtained by:

1. Dense rank;
2. SPLADE rank;
3. locked equal reciprocal-rank fusion; and
4. validation-selected weighted reciprocal-rank fusion?

This is Package A's rank-only first stage. Distance, PPR, path/connectivity,
RRF-plus-structure, and linear-QLS controls require graph computation and will
be frozen separately as Stage A2.

## Frozen RRF rules

For one-indexed source rank `r` and constant `k=60`:

```text
equal_RRF(d) = 0.5/(60 + dense_rank(d))
             + 0.5/(60 + splade_rank(d))
```

Missing-source contributions are zero. Equal score ties are broken by ascending
global node ID.

Weighted RRF searches only the following Dense weights on validation:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

The SPLADE weight is `1 - dense_weight`. Selection uses validation Recall@5.
Ties choose the value closest to equal weighting, then the smaller Dense weight.
Only the selected weight is evaluated on test; unselected weighted test cells
must never be computed or reported.

## Data and metrics

The six fixed datasets are 2Wiki, MuSiQue, WebQSP, HotpotQA, SQuAD, and MetaQA.
Candidate identities, ordering, splits, and golds are unchanged. Recall uses all
query golds as the denominator; conditional metrics use only in-pool golds.

Report candidate ceiling, candidate availability, R@1/R@5/R@20, MRR,
FullCov@20, and conditional R@1/R@5/R@20/MRR. Save compact query-level arrays so
later comparisons remain paired.

## Leakage and stopping rules

- No retraining, graph loading, or candidate regeneration.
- No test-selected RRF weight or constant.
- No structural-control design based on Stage A1 test outcomes.
- No result is spliced into the sealed QLS confirmation table.
- Any Stage A2 structural control requires a new frozen protocol first.

The first stopping point is a validated implementation, one result file per
dataset, a compiled six-dataset table, and an explicit decision about Stage A2.
