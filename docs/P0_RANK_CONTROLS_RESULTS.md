# P0 A1 Dense/SPLADE rank-control results

Status: **complete on all six frozen datasets**.

These are deterministic, training-free controls over the unchanged Dense-top-200 union SPLADE-top-200 candidate contract. Weighted RRF was selected by validation R@5 only. The test split was evaluated only for Dense, SPLADE, locked equal RRF, and the single selected weight.

## Primary R@5 result

| Dataset | Dense | SPLADE | Equal RRF | Selected Dense weight | Selected RRF | Gain over best single |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 64.03 | 67.75 | 68.18 | 0.25 | 68.48 | +0.73 |
| musique_clean | 66.60 | 63.56 | 69.04 | 0.75 | 69.24 | +2.64 |
| webqsp | 8.75 | 9.60 | 10.20 | 0.50 | 10.20 | +0.60 |
| hotpotqa_clean | 70.15 | 69.80 | 72.18 | 0.75 | 72.24 | +2.08 |
| squad_clean | 87.90 | 88.22 | 89.14 | 0.75 | 89.31 | +1.09 |
| metaqa | 8.22 | 13.75 | 10.22 | 0.00 | 13.75 | +0.00 |

Validation-selected RRF improves on the stronger single ranker on five datasets and ties it on MetaQA, where validation selects Dense weight 0.0 (pure SPLADE). The selected weights vary substantially by dataset, so an equal-fusion assumption is not universally optimal.

## Selected rank-control metrics

| Dataset | Candidate ceiling | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 79.67 | 42.05 | 68.48 | 72.30 | 96.71 | 42.40 |
| musique_clean | 94.05 | 37.99 | 69.24 | 81.07 | 89.82 | 59.45 |
| webqsp | 49.06 | 4.80 | 10.20 | 18.53 | 10.50 | 15.72 |
| hotpotqa_clean | 92.95 | 42.39 | 72.24 | 81.45 | 89.47 | 65.21 |
| squad_clean | 99.49 | 72.03 | 89.31 | 94.37 | 79.67 | 94.37 |
| metaqa | 33.51 | 2.67 | 13.75 | 16.30 | 10.67 | 13.94 |

## Descriptive comparison with the frozen learned models

The three learned columns are five-seed means from the already sealed fairness confirmation. These differences are descriptive cross-artifact comparisons, not new confirmatory significance tests.

| Dataset | Selected RRF | Seed-only MLP | QLS-MLP | Seed-aware GNN | RRF − QLS | RRF − GNN |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 68.48 | 65.83 | 68.40 | 69.85 | +0.08 | -1.36 |
| musique_clean | 69.24 | 80.08 | 80.28 | 81.24 | -11.04 | -12.01 |
| webqsp | 10.20 | 29.26 | 33.37 | 33.09 | -23.17 | -22.90 |
| hotpotqa_clean | 72.24 | 73.43 | 77.13 | 77.66 | -4.89 | -5.43 |
| squad_clean | 89.31 | 89.31 | 89.23 | 89.33 | +0.08 | -0.02 |
| metaqa | 13.75 | 23.25 | 30.11 | 30.13 | -16.37 | -16.38 |

## Interpretation

Rank fusion is a necessary control, not the paper's replacement model. It already matches QLS-MLP R@5 within 0.08 points on 2Wiki and SQuAD, which means those two datasets cannot by themselves establish a fixed-structure mechanism. In contrast, QLS-MLP exceeds selected RRF by 4.89 points on HotpotQA, 11.04 on MuSiQue, 16.36 on MetaQA, and 23.17 on WebQSP. Those are the regimes where learned semantic interaction and/or query-local graph summaries add material value beyond rank fusion.

The candidate ceiling is identical for every rank-only method because all methods rerank the same frozen union. These controls cannot repair missing gold nodes.

## Leakage and systems audit

- No node or query embeddings, graph edges, partitions, or model checkpoints are loaded by the rank-control evaluator.
- MetaQA entity identity is restored from the frozen local SPLADE `id_to_idx` bijection; the sparse SPLADE matrix and graph are not used for scoring.
- Per-query metric arrays and SHA-256 source fingerprints are retained under `outputs/p0_rank_controls/`.
- The recorded seconds measure offline artifact evaluation, including source fingerprinting. They are not batch-1 service latency and must not be compared with QLS/GNN online latency.

## Next frozen boundary

A1 answers the semantic rank-fusion control. The next experiment must separately freeze structural-only and linear-combination controls before any of their test results are computed: PPR, distance, path/connectivity, RRF plus structure, and a linear QLS control. No completed QLS/GNN architecture may be tuned against A1.
