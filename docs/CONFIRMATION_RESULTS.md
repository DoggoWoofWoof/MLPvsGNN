# Five-seed lightweight-operator confirmation

> **Historical stage:** this document records the pre-SA plain/Offset phase.
> References to a “topology-free model” apply to the plain or Offset models in
> this experiment, not to the later graph-aware SA-MLP. The current project
> finding is frozen in `SA_MLP_CONFIRMATION_RESULTS.md`.

> **Status:** `CONFIRMATION_GATE_NOT_PAPER_FINAL`  
> **Run date:** 2026-08-25  
> **Frozen protocol:** commit `2c869a8`  
> **Seeds:** 0, 1, 2, 3, 4  
> **Hardware:** one NVIDIA A10 per dataset

## Gate decision

The surprising plain-MLP R@5 result **replicates on both datasets** under the
frozen five-seed protocol. The smaller-capacity hypothesis does not.

- On 2Wiki, width-64 plain MLP exceeds frozen GAT by a paired +2.077 R@5
  percentage points (95% CI +1.218 to +2.936).
- On MuSiQue, width-64 plain MLP exceeds frozen GCN by +1.103 R@5 points
  (95% CI +0.572 to +1.633).
- MuSiQue plain MLP also improves R@1, R@20, MRR, and FullCov@20. On 2Wiki it
  improves R@1/R@5/MRR but loses 0.677 R@20 and 1.600 FullCov@20 points.
- Validation selects width 64 for plain and Offset on both datasets. Widths 16
  and 32 do not retain enough R@5. This experiment therefore rejects the
  proposed 2-4x parameter-reduction claim for the current formulations.
- Width-64 plain MLP remains about 4.0x faster than GAT on 2Wiki and 4.3x
  faster than GCN on MuSiQue, with 104 and 264 MiB less incremental inference
  memory. Its parameter count is only slightly smaller, not substantially so.

The topology-free result is legitimate but conditional: topology is extra
information, yet the matched plain learner has better early/mid-rank retrieval
under this candidate-level task and training budget. It does not establish
that message passing is universally harmful.

## Five-seed confirmation

Values are mean +/- sample standard deviation in percentage points. Latency is
the mean of each seed's median over five synchronized inference repetitions.
Memory is incremental GPU allocation above frozen embeddings and the model's
base allocation.

| Dataset | Model | Params | R@1 | R@5 | R@20 | MRR | FC@20 | ms/query | GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | plain MLP h64 | 204,928 | 29.10 +/- 0.39 | 50.25 +/- 0.50 | 63.32 +/- 0.42 | 75.33 +/- 0.61 | 31.40 +/- 0.73 | 0.0648 | 39.3 |
| 2Wiki | Offset K=1 h64 | 209,024 | 39.54 +/- 0.13 | 48.41 +/- 0.40 | 54.29 +/- 0.40 | 90.62 +/- 0.20 | 19.81 +/- 0.70 | 0.0748 | 39.3 |
| 2Wiki | Offset K=4 h64 | 221,504 | 39.54 +/- 0.13 | 49.26 +/- 0.23 | 55.25 +/- 0.18 | 90.40 +/- 0.22 | 21.61 +/- 0.48 | 0.0765 | 48.0 |
| 2Wiki | frozen GAT h64 | 213,504 | 26.21 +/- 0.56 | 48.17 +/- 0.47 | 64.00 +/- 0.27 | 70.17 +/- 1.04 | 33.00 +/- 0.54 | 0.2597 | 143.2 |
| MuSiQue | plain MLP h64 | 204,928 | 37.31 +/- 0.11 | 76.72 +/- 0.07 | 87.12 +/- 0.19 | 89.09 +/- 0.21 | 72.22 +/- 0.26 | 0.0633 | 37.7 |
| MuSiQue | Offset K=1 h64 | 209,024 | 38.08 +/- 0.15 | 70.74 +/- 0.26 | 83.03 +/- 0.34 | 89.21 +/- 0.22 | 65.11 +/- 0.49 | 0.0762 | 37.7 |
| MuSiQue | Offset K=4 h64 | 221,504 | 38.04 +/- 0.10 | 69.90 +/- 0.33 | 82.05 +/- 0.17 | 88.96 +/- 0.21 | 63.39 +/- 0.42 | 0.0791 | 45.9 |
| MuSiQue | frozen GCN h64 | 209,216 | 36.49 +/- 0.33 | 75.62 +/- 0.38 | 86.78 +/- 0.23 | 87.94 +/- 0.46 | 71.74 +/- 0.46 | 0.2722 | 301.4 |

## Paired difference from the frozen GNN

Differences are percentage points, averaged over within-seed pairs. Intervals
are two-sided 95% t intervals with four degrees of freedom. Positive favors the
topology-free model.

| Dataset | Model | dR@1 [95% CI] | dR@5 [95% CI] | dR@20 [95% CI] | dMRR [95% CI] | dFC@20 [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | plain h64 | +2.893 [+2.476,+3.310] | +2.077 [+1.218,+2.936] | -0.677 [-1.121,-0.233] | +5.159 [+4.272,+6.046] | -1.600 [-1.984,-1.216] |
| 2Wiki | Offset K=1 | +13.337 [+12.727,+13.947] | +0.243 [-0.130,+0.617] | -9.710 [-10.025,-9.395] | +20.448 [+19.164,+21.731] | -13.187 [-13.681,-12.692] |
| 2Wiki | Offset K=4 | +13.330 [+12.543,+14.117] | +1.093 [+0.589,+1.598] | -8.750 [-9.013,-8.487] | +20.230 [+18.734,+21.726] | -11.387 [-12.128,-10.646] |
| MuSiQue | plain h64 | +0.815 [+0.448,+1.182] | +1.103 [+0.572,+1.633] | +0.339 [+0.258,+0.421] | +1.153 [+0.544,+1.762] | +0.481 [+0.091,+0.871] |
| MuSiQue | Offset K=1 | +1.584 [+1.179,+1.990] | -4.877 [-5.594,-4.161] | -3.744 [-4.072,-3.415] | +1.274 [+0.790,+1.759] | -6.627 [-7.115,-6.138] |
| MuSiQue | Offset K=4 | +1.553 [+1.130,+1.976] | -5.721 [-6.429,-5.013] | -4.732 [-5.182,-4.281] | +1.021 [+0.322,+1.720] | -8.351 [-9.109,-7.593] |

## Capacity sweep and validation-only selection

| Dataset | Family | h16 val/test R@5 | h32 val/test R@5 | h64 val/test R@5 | Selected |
|---|---|---:|---:|---:|---|
| 2Wiki | plain | 26.62 / 26.13 | 39.62 / 39.61 | 50.45 / 50.25 | h64, 204,928 params |
| 2Wiki | Offset K=1 | 45.48 / 44.57 | 47.91 / 46.90 | 49.39 / 48.41 | h64, 209,024 params |
| MuSiQue | plain | 45.47 / 46.30 | 63.80 / 64.41 | 75.90 / 76.72 | h64, 204,928 params |
| MuSiQue | Offset K=1 | 58.19 / 58.81 | 65.64 / 66.35 | 70.06 / 70.74 | h64, 209,024 params |

The capacity curves are monotone and the h64 validation margins greatly exceed
the preregistered 0.1-point tie margin. No smaller model is eligible through a
tie-break. Test values are shown for transparency but were not used to select.

## Coverage mechanism diagnosis

The Offset-minus-GNN R@20 gap becomes monotonically more negative as the gold
set grows in both datasets:

| Dataset | Model | 2 gold | 3 gold | 4 gold | Monotonic loss? |
|---|---|---:|---:|---:|---|
| 2Wiki | Offset K=1 | -9.10 | n/a | -12.00 | yes |
| 2Wiki | Offset K=4 | -8.24 | n/a | -10.65 | yes |
| MuSiQue | Offset K=1 | -2.61 | -6.62 | -7.07 | yes |
| MuSiQue | Offset K=4 | -3.49 | -7.87 | -8.36 | yes |

The pattern is not just R@20. On MuSiQue, K=1 Offset's R@5 gap changes from
-3.52 points at two golds to -8.01 at three and -9.84 at four. Its R@1 advantage
changes from +2.69 to -1.14 and -1.88. This supports a multi-target coverage
failure rather than a uniformly inferior representation.

MuSiQue semantic hop count was recovered for all 1,995 test queries from raw
question-decomposition length, but it is perfectly collinear with gold-document
count in this substrate. Hop and answer-multiplicity effects therefore cannot
be separately identified. 2Wiki has no per-query hop label in its frozen or
processed records; no hop labels were invented.

## Interpretation and historical next gate

The result supports a stronger, narrower paper direction:

> At matched capacity, message passing is not required for strong candidate
> retrieval on 2Wiki and MuSiQue; a topology-free MLP improves R@5 at roughly
> four times lower operator latency. Relational Offset models specialize in
> early-rank precision but fail increasingly as the relevant set grows.

This is still not a final NeurIPS claim. It is two datasets, one frozen training
budget, and candidate-level rather than end-to-end retrieval.

The preregistered condition for one coverage-aware Offset variant was satisfied.
That exact variant was subsequently frozen and evaluated; it failed its primary
criterion on both datasets, as recorded in `COVERAGE_VARIANT_RESULTS.md`.
Generic topology perturbations remain deferred and are not currently scheduled.

Raw model results and reproducible stratified analyses are retained under
`outputs/confirmation/`.
