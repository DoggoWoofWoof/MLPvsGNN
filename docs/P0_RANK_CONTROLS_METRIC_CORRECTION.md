# P0 A1 MRR scope correction

Status: **frozen corrective action after A1 v1 and before A2 test evaluation**.

During implementation of A2's full-ranking reproduction gate, inspection found
that the A1 evaluator passed only the top 20 ranked items to the metric
function. This is sufficient for Recall@1/5/20 and FullCov@20, but it silently
made the reported `mrr` an MRR@20. The sealed learned-model evaluation computes
MRR over the full candidate ranking.

## Locked correction

The A1 evaluator will be rerun with:

- Dense ranked over its full frozen top-200 list;
- SPLADE ranked over its full frozen top-200 list; and
- RRF ranked over the full unique Dense/SPLADE candidate union.

No candidate, label, split, validation weight, RRF constant, tie rule, or
selection rule changes. Recall@1/5/20, FullCov@20, candidate ceiling, and
candidate availability must reproduce v1. Only MRR and conditional MRR may
increase when a query's first retrieved gold is below rank 20.

The v1 MRR cells are withdrawn and must not be used. Corrected artifacts and a
corrected report receive a new result version. The defect and correction remain
documented rather than rewriting history.

## A2 consequence

A2 has not accessed test results. Its selected-RRF reproduction gate will use
the corrected full-ranking A1 metrics. A separate A2 compatibility amendment
will point from the originally named A1 v1 result tag to the corrected v2 tag;
all A2 structural rules remain unchanged.
