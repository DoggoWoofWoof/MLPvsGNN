# Package E1: Validation-Only Phase Screen

Status: **frozen before the validation sweep**. This stage computes no test
metrics and does not train a crossover predictor.

## Purpose

The clean table and Package B/C curves are observational anchors. This screen
causally varies two core quantities while holding architecture and supervision
fixed:

```text
candidate-topology quality
raw node-feature quality
```

The goal is to locate candidate crossover regions where `GNN - QLS` validation
R@5 changes sign. Five-seed test confirmation is a later frozen stage.

## Locked axes

All topology interventions operate on the exact candidate-induced query graph
after the frozen candidates have been selected. They use no relevance labels.
This scope is intentional: uniform random edges in a million-node global graph
would almost never land twice inside a 50--400 node candidate pool and would
therefore be a weak intervention on the graph actually consumed by the models.

| Axis | Rates | Operation |
|---|---|---|
| degree-preserving rewiring | 10/25/50/100% | permute destinations within each query graph while preserving directed in/out degree with multiplicity |
| random-edge addition | 10/25/50/100% of stored local edges | append deterministic uniform non-self directed edges; retain every clean edge |
| hub injection | 10/25/50/100% | redirect selected destinations toward the highest-degree local node while preserving edge count and source degree |
| raw node-feature masking | 25/50/75/100% | zero deterministic scalar entries in frozen node embeddings; query embeddings and graph stay clean |

For topology axes, QLS query-local distance/path/PPR features and GNN
adjacency are rebuilt from the same perturbed packed graph. Corpus-static QLS
features remain frozen to the clean global graph, making this a local topology
quality intervention rather than a corpus rebuild. For feature masking, QLS
and GNN receive the exact same masked raw node matrix; topology and fixed
structural summaries remain clean.

## Screening and confirmation boundary

The screen trains seed 0 only and evaluates validation only. The rate-zero
reference comes from the sealed confirmation's selected validation checkpoint.
No test per-query array or aggregate is computed.

For each dataset and axis:

1. if adjacent rates bracket a sign change, retain the bracket, clean point,
   and axis endpoint;
2. if no sign changes, retain only clean and the endpoint;
3. if a rate is exactly zero-gap, retain it, adjacent rates, and endpoints.

The union of selected rates across datasets is frozen per axis. Only then are
five canonical seeds trained and test metrics opened for those rates. This
prevents perturbation levels from being tuned to manufacture an MLP or GNN win.

The predictor remains prohibited until the confirmed regimes contain clear
help, neutral, and harm regions. All screening cells remain publishable in the
appendix as validation-only exploration; none is silently discarded.


## Required headroom companion

Every screen axis perturbs the graph or the raw node features. None of them
touches the candidate pool, so the candidate ceiling
(`CANDIDATE_HEADROOM_PROTOCOL.md`) is **constant across every cell of the
screen** for a given dataset, and identical for both models within a cell.

That is a useful invariant rather than a caveat: any change in validation
metric across perturbation levels is attributable to the intervention and the
model, and cannot be an artifact of candidate generation. The
`validation_gnn_minus_qls` contrast is likewise ceiling-free.

The ceiling is still needed when reading absolute levels across datasets. A cell
on MetaQA is scored against a far lower ceiling than the same cell on SQuAD, so
absolute robustness levels must not be compared across datasets without it.

Crossover and end-point rates are selected by the locked rule on validation
values alone. Headroom numbers may not enter that selection, and no test metric
may be consulted to choose a rate.
