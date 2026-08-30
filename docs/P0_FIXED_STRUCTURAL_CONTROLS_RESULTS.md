# P0 A2 fixed structural-control results

Status: **complete on all six frozen datasets**.

Every method was locked before test access and uses the exact sealed QLS query-local feature cache. There are no learned parameters, seeds, A2 validation choices, or message-passing operations.

## Complete R@5 table

| Dataset | Selected RRF | Distance | PPR | Path/connectivity | Structural summary | RRF + PPR | RRF + summary |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 68.48 | 45.63 | 42.52 | 29.03 | 33.02 | 65.73 | 59.23 |
| musique_clean | 69.24 | 54.22 | 41.96 | 22.49 | 31.75 | 67.46 | 54.27 |
| webqsp | 10.20 | 14.49 | 15.05 | 14.16 | 15.39 | 10.98 | 10.83 |
| hotpotqa_clean | 72.24 | 58.53 | 59.97 | 48.68 | 56.10 | 73.24 | 68.30 |
| squad_clean | 89.31 | 65.04 | 63.31 | 17.91 | 50.22 | 88.20 | 76.05 |
| metaqa | 13.75 | 10.10 | 16.10 | 16.34 | 18.16 | 13.77 | 14.12 |

## Structural signal and remaining learned-model gap

The `best` labels below summarize the fully reported table; they are descriptive test maxima and are not selected models or inputs to A3.

| Dataset | Best structural-only | Δ vs RRF | Best training-free | Best training-free R@5 | Δ vs QLS | Δ vs GNN |
|---|---|---:|---|---:|---:|---:|
| 2wiki_clean | distance | -22.85 | selected_rrf | 68.48 | +0.08 | -1.36 |
| musique_clean | distance | -15.02 | selected_rrf | 69.24 | -11.04 | -12.01 |
| webqsp | structural_summary | +5.20 | structural_summary | 15.39 | -17.97 | -17.70 |
| hotpotqa_clean | ppr | -12.27 | selected_rrf_plus_ppr | 73.24 | -3.89 | -4.42 |
| squad_clean | distance | -24.27 | selected_rrf | 89.31 | +0.08 | -0.02 |
| metaqa | structural_summary | +4.41 | structural_summary | 18.16 | -11.96 | -11.98 |

## Result

Fixed query-local structure contains real retrieval signal. Structural summary alone improves over selected RRF by 5.20 R@5 points on WebQSP and 4.41 on MetaQA. Locked RRF+PPR improves HotpotQA by 1.01 points. These are the three relational/graph regimes where the original plain GNN had won.

The simple rules are not a sufficient replacement for QLS-MLP. Relative to the best fully reported training-free method, QLS retains 17.97 points on WebQSP, 11.96 on MetaQA, 11.04 on MuSiQue, and 3.89 on HotpotQA. On 2Wiki and SQuAD, rank fusion alone already matches QLS within 0.08 points.

Naive equal fusion is also not universally beneficial: adding structural rankings damages 2Wiki, MuSiQue, and SQuAD. The next legitimate control is therefore a tiny linear model trained only on train labels and selected on validation—not a new deep architecture and not a test-tuned fusion weight.

## Audit

- All six candidate-order/source/feature contracts passed exact SHA-256 alignment.
- The selected-RRF reference reproduced corrected A1 with zero aggregate difference.
- No graph or node/query embedding was loaded by A2; only frozen rank arrays, labels/splits for evaluation, and memory-mapped structural scalars were read.
- Modal used CPU workers because this stage contains only fixed scalar scoring and sorting. GPUs remain reserved for the learned A3 control.
- Reported runtime is offline artifact evaluation, including full candidate hashing and compression; it is not service latency.

## Stopping point

A2 is closed. Do not tune structural formulas or fusion weights against these test results. A3 may proceed only under its own frozen feature, optimizer, validation, seed, and test-access contract.
