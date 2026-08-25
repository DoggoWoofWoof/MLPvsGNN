# Engineering smoke log (not paper evidence)

Date: 2026-08-25

The first end-to-end L2 export and paired-model run used the existing read-only
WebQSP signal cache. The artifact was stamped `pilot_test_only`; the runner
re-split 159 cached test queries solely to exercise the code path.

Configuration:

- 111/24/24 engineering split;
- one seed;
- one epoch;
- one-layer parameter-matched MLP/GCN pair;
- CPU;
- untyped CRAG graph;
- incomplete WebQSP cache;
- no valid hyperparameter selection.

Observed engineering metrics:

| Metric | MLP | GCN |
|---|---:|---:|
| R@1 | 25.17 | 21.64 |
| R@5 | 37.19 | 39.04 |
| R@20 | 52.11 | 70.96 |
| MRR | 0.481 | 0.452 |
| inference ms/query | 3.52 | 18.60 |
| candidate ceiling | 85.97 | 85.97 |

Interpretation: the exporter, induced-graph builder, gradient audit, training
loop, and metric path execute successfully. The result is intentionally not a
scientific comparison. It also demonstrates why the eventual claim cannot be
“MLP always wins”: even this crude smoke run changes winner with retrieval
budget, while the MLP retains a large latency advantage. Canonical splits,
multiple seeds, validation, typed edges, and full sweeps are required.

This one-epoch CPU run remains a smoke test even after the later three-dataset
Modal pilot. In particular, its WebQSP R@1/R@5 winner reversal is retained as a
warning against collapsing the research question into one metric; it is not a
baseline or a number to report in the paper. See `docs/PILOT3_RESULTS.md` for
the subsequent protocol audit.
