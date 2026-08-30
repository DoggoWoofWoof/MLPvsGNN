# P0 A2 corrected A1 reference amendment

Status: **frozen before A2 test evaluation**.

The original A2 protocol named `p0-rank-controls-results-v1` as its semantic
reference. Before any A2 test result was computed, A1's truncated-MRR defect
was found and corrected under `P0_RANK_CONTROLS_METRIC_CORRECTION.md`.

A2 therefore uses `p0-rank-controls-results-v2`. This changes only the A1 MRR
and conditional-MRR reference cells from MRR@20 to full-ranking MRR. A1
candidate sets, rankings, validation-selected weights, R@1/5/20, FullCov@20,
ceilings, and all A2 structural methods remain unchanged.

The A2 selected-RRF reproduction gate must match the v2 full-ranking metrics.
