# Complete-data relational-operator screen

> **Status:** `SCREENING_ONLY_NOT_PAPER_FINAL`  
> **Run date:** 2026-08-25  
> **Frozen implementation:** commit `5a6b8e3`  
> **Seed/budget:** seed 0, three epochs, hidden width 64, one GNN layer  
> **Hardware:** one NVIDIA A10 per dataset, datasets run in parallel

## Decision

The result is **mixed, not an Offset-MLP win**.

- The operator-level efficiency result is strong in this screen. Single-offset
  inference is 3.16x-5.45x faster than the validation-selected GNN and uses
  104-264 MiB less incremental GPU memory. Model-training time is 1.38x-2.06x
  shorter.
- The intended parameter claim is not supported: the single Offset-MLP is only
  0.09%-2.10% smaller than the selected GNN because this screen deliberately
  made them nearly parameter-matched. The defensible claim is lower compute and
  memory at similar parameter count, not "substantially fewer parameters."
- Quality is metric- and dataset-dependent. On 2Wiki, single Offset improves
  R@1 by 12.48 points and MRR by 18.65 points versus validation-selected GAT,
  but loses 0.18 R@5 and 10.12 R@20 points. K=4 recovers a small +0.57 R@5
  advantage but still loses 8.77 R@20 points.
- On MuSiQue, single Offset gains 1.18 R@1 and 0.89 MRR points versus the
  validation-selected GCN but loses 4.42 R@5 and 3.81 R@20 points. On WebQSP,
  it is essentially tied at R@1 and loses at R@5, R@20, and MRR.
- K=4 is not consistently better than K=1. It hurts WebQSP and MuSiQue and does
  not resolve the loss in multi-gold coverage on 2Wiki.
- Plain MLP has the highest test R@5 in 2Wiki and MuSiQue in this seed. That is
  useful evidence against assuming that either relational translation or
  message passing is always necessary, but it is not a paper claim yet.

The current evidence suggests a **top-rank sharpness versus multi-answer
coverage tradeoff**. Offset translation often puts one relevant item first,
especially on 2Wiki, but retrieves fewer of the remaining relevant items by
rank 20. Five-seed confirmation and ablations are needed before treating that
as a mechanism rather than a seed/budget effect.

No follow-up experiment was launched after this table.

## Main screening table

All effectiveness values are unconditional percentages over the complete test
split. `FC@20` is full coverage at 20. `GPU MiB` is incremental operator memory
followed by total allocated memory in parentheses. Latency is synchronized
operator scoring with candidates and induced topology already prepared; it is
not end-to-end retrieval latency. All reported parameters are trainable because
the embedding stores are frozen external inputs.

| Dataset | Model | R@1 | R@5 | R@20 | MRR | FC@20 | Params | Latency ms/q | Queries/s | GPU MiB inc. (total) | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WebQSP | plain MLP | 13.53 | 27.66 | 38.27 | 34.46 | 28.93 | 204,928 | 0.0668 | 14,971 | 38.4 (4,693.2) | 2.04 |
| WebQSP | Offset-MLP | 15.31 | 27.95 | 39.57 | 36.24 | 30.82 | 209,024 | 0.0766 | 13,054 | 38.4 (4,693.2) | 1.40 |
| WebQSP | Offset-MLP K=4 | 9.47 | 26.52 | 38.91 | 31.92 | 30.19 | 221,504 | 0.0780 | 12,814 | 46.7 (4,701.6) | 1.42 |
| WebQSP | GCN | 16.70 | 29.91 | 40.35 | 40.31 | 31.45 | 209,216 | 0.3357 | 2,979 | 275.8 (4,930.6) | 2.31 |
| WebQSP | GraphSAGE | 16.75 | 26.94 | 39.52 | 37.33 | 30.82 | 213,312 | 0.2858 | 3,500 | 235.4 (4,890.3) | 2.15 |
| WebQSP | GAT | 15.26 | 31.10 | 40.65 | 36.99 | 30.82 | 213,504 | 0.4174 | 2,396 | 298.9 (4,953.8) | 2.87 |
| WebQSP | GIN | 14.40 | 30.70 | 40.25 | 38.30 | 30.82 | 213,376 | 0.2727 | 3,667 | 235.3 (4,890.3) | 2.08 |
| 2Wiki | plain MLP | 29.53 | 50.37 | 62.68 | 76.09 | 30.20 | 204,928 | 0.0628 | 15,919 | 39.3 (578.8) | 20.49 |
| 2Wiki | Offset-MLP | 39.60 | 48.75 | 53.62 | 90.59 | 18.93 | 209,024 | 0.0771 | 12,975 | 39.3 (578.9) | 21.27 |
| 2Wiki | Offset-MLP K=4 | 39.45 | 49.50 | 54.97 | 90.19 | 21.33 | 221,504 | 0.0766 | 13,047 | 48.0 (587.7) | 21.07 |
| 2Wiki | GCN | 26.02 | 47.68 | 62.57 | 70.37 | 30.20 | 209,216 | 0.2126 | 4,703 | 132.4 (672.0) | 27.53 |
| 2Wiki | GraphSAGE | 26.97 | 48.33 | 62.62 | 71.36 | 30.60 | 213,312 | 0.1572 | 6,362 | 100.7 (640.3) | 26.42 |
| 2Wiki | GAT | 27.12 | 48.93 | 63.73 | 71.94 | 32.20 | 213,504 | 0.2438 | 4,102 | 143.2 (683.0) | 31.13 |
| 2Wiki | GIN | 25.48 | 48.17 | 63.00 | 69.82 | 31.40 | 213,376 | 0.1461 | 6,846 | 100.6 (640.4) | 25.50 |
| MuSiQue | plain MLP | 37.41 | 76.75 | 87.14 | 89.30 | 72.28 | 204,928 | 0.0640 | 15,626 | 37.7 (300.2) | 24.67 |
| MuSiQue | Offset-MLP | 38.16 | 71.13 | 83.09 | 89.47 | 65.31 | 209,024 | 0.0731 | 13,678 | 37.7 (300.2) | 25.80 |
| MuSiQue | Offset-MLP K=4 | 37.96 | 69.38 | 81.89 | 88.74 | 63.21 | 221,504 | 0.0740 | 13,510 | 45.9 (308.5) | 26.38 |
| MuSiQue | GCN | 36.98 | 75.55 | 86.89 | 88.58 | 71.68 | 209,216 | 0.2627 | 3,806 | 302.0 (564.5) | 35.73 |
| MuSiQue | GraphSAGE | 36.98 | 76.02 | 86.65 | 88.39 | 71.58 | 213,312 | 0.2113 | 4,732 | 261.9 (524.5) | 33.06 |
| MuSiQue | GAT | 37.11 | 76.35 | 86.99 | 88.69 | 72.38 | 213,504 | 0.3214 | 3,111 | 328.4 (591.1) | 42.22 |
| MuSiQue | GIN | 36.67 | 75.60 | 86.54 | 87.99 | 70.63 | 213,376 | 0.2035 | 4,914 | 261.8 (524.6) | 34.22 |

Topology preprocessing required 31.26 s for WebQSP, 191.91 s for 2Wiki, and
383.89 s for MuSiQue. It constructs candidate-induced graphs once and is not
included in per-model training or inference time. Offset/plain models do not
require this preprocessing. The implementation is a correctness-first Python
construction, so these numbers characterize the current pipeline rather than
an optimized graph sampler.

## Candidate-conditional retrieval

WebQSP has in-pool gold for 63.52% of test queries and a mean test candidate
ceiling of 49.06%. 2Wiki and MuSiQue test availability is 100%; their mean
ceilings are 79.67% and 94.05%. Conditional recall divides by the gold items
that are actually present in the common pool and excludes queries with no
in-pool gold. `cHit@5` asks whether at least one such item occurs by rank 5.

| Dataset | Model | cR@1 | cR@5 | cR@20 | cHit@5 | cMRR |
|---|---|---:|---:|---:|---:|---:|
| WebQSP | plain MLP | 28.64 | 59.79 | 80.14 | 71.29 | 54.25 |
| WebQSP | Offset-MLP | 32.08 | 58.85 | 80.89 | 72.28 | 57.05 |
| WebQSP | Offset-MLP K=4 | 25.35 | 56.58 | 80.14 | 69.31 | 50.25 |
| WebQSP | GCN | 35.53 | 64.24 | 82.57 | 77.23 | 63.46 |
| WebQSP | GraphSAGE | 34.18 | 58.98 | 82.07 | 71.29 | 58.77 |
| WebQSP | GAT | 31.82 | 66.21 | 83.85 | 80.20 | 58.23 |
| WebQSP | GIN | 34.39 | 65.13 | 82.46 | 79.21 | 60.29 |
| 2Wiki | plain MLP | 40.31 | 65.94 | 80.74 | 89.67 | 76.09 |
| 2Wiki | Offset-MLP | 53.51 | 63.52 | 69.19 | 92.67 | 90.59 |
| 2Wiki | Offset-MLP K=4 | 53.42 | 64.37 | 70.97 | 92.27 | 90.19 |
| 2Wiki | GCN | 35.31 | 62.79 | 80.62 | 87.53 | 70.37 |
| 2Wiki | GraphSAGE | 37.07 | 63.26 | 80.66 | 86.73 | 71.36 |
| 2Wiki | GAT | 36.88 | 63.84 | 81.74 | 86.60 | 71.94 |
| 2Wiki | GIN | 34.82 | 62.85 | 80.71 | 86.53 | 69.82 |
| MuSiQue | plain MLP | 40.59 | 81.51 | 92.57 | 95.89 | 89.30 |
| MuSiQue | Offset-MLP | 41.78 | 75.76 | 88.15 | 95.09 | 89.47 |
| MuSiQue | Offset-MLP K=4 | 41.48 | 73.83 | 86.90 | 94.19 | 88.74 |
| MuSiQue | GCN | 39.96 | 80.35 | 92.45 | 95.89 | 88.58 |
| MuSiQue | GraphSAGE | 39.97 | 80.74 | 91.98 | 95.59 | 88.39 |
| MuSiQue | GAT | 40.15 | 81.07 | 92.39 | 95.64 | 88.69 |
| MuSiQue | GIN | 39.51 | 80.41 | 92.00 | 95.64 | 87.99 |

## Exact Offset minus validation-selected best GNN

The comparator is selected using validation R@5 only: GAT for WebQSP, GAT for
2Wiki, and GCN for MuSiQue. It is not re-selected per test metric. Effectiveness
deltas are percentage points; positive favors Offset. Memory savings are
positive MiB reductions in incremental inference allocation. Parameter delta is
Offset minus GNN, so negative means Offset is smaller.

| Dataset | Offset | Best GNN (val R@5) | dR@1 | dR@5 | dR@20 | dMRR | Latency speedup | Memory saved MiB | Param delta | Train speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WebQSP | K=1 | GAT (30.864%) | +0.0526 | -3.1523 | -1.0756 | -0.7487 | 5.448x | 260.5 | -4,480 | 2.057x |
| WebQSP | K=4 | GAT (30.864%) | -5.7875 | -4.5816 | -1.7341 | -5.0660 | 5.348x | 252.2 | +8,000 | 2.019x |
| 2Wiki | K=1 | GAT (49.025%) | +12.4833 | -0.1833 | -10.1167 | +18.6495 | 3.163x | 103.9 | -4,480 | 1.464x |
| 2Wiki | K=4 | GAT (49.025%) | +12.3333 | +0.5667 | -8.7667 | +18.2546 | 3.180x | 95.3 | +8,000 | 1.478x |
| MuSiQue | K=1 | GCN (75.188%) | +1.1779 | -4.4236 | -3.8053 | +0.8928 | 3.593x | 264.3 | -192 | 1.385x |
| MuSiQue | K=4 | GCN (75.188%) | +0.9816 | -6.1738 | -5.0042 | +0.1627 | 3.549x | 256.1 | +12,288 | 1.355x |

## Integrity and limitations

- All models within a dataset used one shared candidate, label, and split
  contract. The result JSON records SHA-256 hashes for those contracts and all
  six frozen source artifacts.
- Complete canonical splits were used: WebQSP 1,104/315/159, 2Wiki
  10,500/3,000/1,500, and MuSiQue 13,956/3,987/1,995.
- GNNs received only candidate-induced adjacency as privileged information;
  Offset models never read adjacency. No C-RAG model or fusion code was
  imported.
- This is one screening seed under a uniform small budget, not a tuned,
  compute-matched, or statistically confirmed comparison.
- Latency excludes candidate generation, host-to-device loading, and one-time
  graph preprocessing. Peak memory includes the complete frozen embedding
  tables; incremental memory better isolates operator cost.
- The result can motivate a frozen five-seed confirmation but cannot support a
  NeurIPS claim by itself.

Raw results are retained in `outputs/operator_screen/{webqsp,2wiki_clean,musique_clean}.json`.
