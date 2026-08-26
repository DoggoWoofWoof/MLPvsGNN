# Six-dataset MLP vs message-passing results

> **Historical clean-comparison stage:** this table established that the plain
> MLP/GNN boundary is dataset-dependent and motivated SA-MLP. “Topology-free”
> below refers only to the plain MLP. The final graph-aware,
> non-message-passing comparison is in `SA_MLP_CONFIRMATION_RESULTS.md`.

All values are five-seed means. Deltas are paired MLP minus GNN percentage points.

## Stop-gate conclusion

The frozen R@5 comparison yields two MLP wins whose paired five-seed 95% intervals exclude zero (2Wiki and MuSiQue), three GNN wins (WebQSP, HotpotQA, and MetaQA), and one neutral result (SQuAD). This rejects both universal claims: neither message passing nor a topology-free MLP dominates every retrieval regime.

The MLP is 3.64--9.92x faster and saves 104--2431 MiB of incremental peak GPU memory. Parameter counts are effectively matched: the GNNs have only 1.021--1.042x the MLP parameters. The result is therefore a latency/memory tradeoff, not a claim of materially fewer MLP parameters.

MetaQA does not show an increasing GNN advantage with hop count: its R@5 advantage is 25.53 points at 1 hop, 2.39 at 2 hops, and 0.85 at 3 hops. Hop count alone is not the mechanism. The causal reason for the cross-dataset boundary remains untested at this stop gate and must not be inferred from dataset names.

## Main effectiveness table

| Dataset | GNN | MLP R@1 | GNN R@1 | ΔR@1 | MLP R@5 | GNN R@5 | ΔR@5 | ΔR@20 | ΔMRR | ΔFullCov@20 | R@5 paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | GAT | 29.10 | 26.21 | 2.89 | 50.25 | 48.17 | 2.08 | -0.68 | 5.16 | -1.60 | [1.22, 2.94] |
| musique_clean | GCN | 37.31 | 36.49 | 0.81 | 76.72 | 75.62 | 1.10 | 0.34 | 1.15 | 0.48 | [0.57, 1.63] |
| webqsp | GAT | 13.71 | 16.15 | -2.44 | 26.27 | 30.32 | -4.05 | -2.27 | -3.87 | -2.14 | [-4.72, -3.38] |
| hotpotqa_clean | GIN | 32.05 | 32.51 | -0.46 | 57.48 | 62.12 | -4.64 | -5.23 | -1.34 | -8.82 | [-5.11, -4.17] |
| squad_clean | GCN | 50.00 | 50.01 | -0.01 | 74.02 | 74.03 | -0.01 | -0.51 | -0.11 | -0.51 | [-0.27, 0.25] |
| metaqa | GAT | 9.34 | 14.86 | -5.52 | 17.73 | 25.44 | -7.71 | -5.23 | -8.13 | -5.26 | [-7.80, -7.63] |

## Candidate-conditional table

| Dataset | Candidate ceiling | Queries with ≥1 in-pool gold | MLP conditional R@5 | GNN conditional R@5 |
|---|---:|---:|---:|---:|
| 2wiki_clean | 79.67 | 100.00 | 65.72 | 62.62 |
| musique_clean | 94.05 | 100.00 | 81.54 | 80.47 |
| webqsp | 49.06 | 63.52 | 57.62 | 65.07 |
| hotpotqa_clean | 92.95 | 99.79 | 62.58 | 67.00 |
| squad_clean | 99.49 | 99.49 | 74.40 | 74.41 |
| metaqa | 33.51 | 51.55 | 51.53 | 69.74 |

## MetaQA hop table

| Hop | Test queries | MLP R@5 | GNN R@5 | ΔR@5 | Paired 95% CI | MLP MRR | GNN MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9947 | 41.16 | 66.69 | -25.53 | [-25.88, -25.18] | 39.86 | 62.64 |
| 2 | 14872 | 8.90 | 11.29 | -2.39 | [-2.49, -2.28] | 12.06 | 15.41 |
| 3 | 14274 | 10.59 | 11.44 | -0.85 | [-0.95, -0.75] | 28.05 | 30.95 |

## Systems table

| Dataset | Nodes | Edges | MLP ms/q | GNN ms/q | GNN/MLP latency | MLP incr. MiB | GNN incr. MiB | Saved MiB | MLP params | GNN params | Cold topology s | Packed topology GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 65865 | 855146 | 0.0648 | 0.2597 | 4.01× | 39.3 | 143.2 | 104.0 | 204928 | 213504 | 185.8 | -- |
| musique_clean | 13672 | 280108 | 0.0633 | 0.2722 | 4.30× | 37.7 | 301.4 | 263.7 | 204928 | 209216 | 391.0 | -- |
| webqsp | 781485 | 13379166 | 0.0634 | 0.4141 | 6.53× | 38.4 | 298.8 | 260.4 | 204928 | 213504 | 0.7 | 0.027 |
| hotpotqa_clean | 507494 | 16223058 | 0.0641 | 0.2335 | 3.64× | 39.4 | 2400.7 | 2361.3 | 204928 | 213376 | 66.3 | 1.457 |
| squad_clean | 19029 | 2857316 | 0.0625 | 0.6202 | 9.92× | 37.0 | 2467.6 | 2430.6 | 204928 | 209216 | 153.7 | 5.989 |
| metaqa | 40151 | 585728 | 0.0623 | 0.2449 | 3.93× | 41.3 | 205.2 | 163.8 | 204928 | 213504 | 169.0 | 2.283 |

## Claim boundary and historical next decision

What is established: under identical frozen features, candidates, labels, loss, splits, seeds, and training budget, adding the validation-selected message-passing model helps on three datasets, hurts on two, and is neutral on one; its inference cost is higher on all six. Candidate-conditional R@5 preserves the same directions, so the boundary is not an artifact of missing candidate-pool golds.

What is not established: these tables do not identify homophily, neighborhood noise, hubness, answer multiplicity, or feature quality as the cause. They also do not support an all-dataset MLP claim, a fewer-parameters claim, or a claim that GNN value grows with query hops.

The preregistered stop condition is satisfied. No topology perturbation, mechanism predictor, Offset rescue, or new architecture has been run as part of this gate.
