# P0 A3 linear rank + structure results

Status: **complete on all six frozen datasets and five canonical seeds**.

A3 is a bias-free 19-parameter linear scorer over two source-rank, seven static-graph, and ten query-local structural features. It loads neither graph adjacency nor node/query embeddings. All learning-rate and epoch choices use validation only.

## Main R@5 decomposition

The `best A2` column is a descriptive maximum from the already complete A2 table; it was not used to select A3 inputs or settings.

| Dataset | Selected RRF | Best A2 | A3 linear | Seed-only MLP | QLS-MLP | Seed-aware GNN |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 68.48 | 68.48 (selected_rrf) | 68.57 ± 0.02 | 65.83 | 68.40 | 69.85 |
| musique_clean | 69.24 | 69.24 (selected_rrf) | 68.51 ± 0.09 | 80.08 | 80.28 | 81.24 |
| webqsp | 10.20 | 15.39 (structural_summary) | 22.21 ± 0.00 | 29.26 | 33.37 | 33.09 |
| hotpotqa_clean | 72.24 | 73.24 (selected_rrf_plus_ppr) | 74.58 ± 0.06 | 73.43 | 77.13 | 77.66 |
| squad_clean | 89.31 | 89.31 (selected_rrf) | 89.50 ± 0.03 | 89.31 | 89.23 | 89.33 |
| metaqa | 13.75 | 18.16 (structural_summary) | 24.48 ± 0.03 | 23.25 | 30.11 | 30.13 |

## What A3 establishes

Learned weighting of fixed rank and structure is genuinely useful, but it is not the whole QLS result. Relative to the best training-free A2 rule, A3 gains +6.82 R@5 points on WebQSP, +6.32 on MetaQA, and +1.34 on HotpotQA. It essentially ties the rank baseline on 2Wiki and SQuAD and loses 0.73 on MuSiQue.

QLS-MLP still leads A3 by 11.77 points on MuSiQue, 11.16 on WebQSP, 5.63 on MetaQA, and 2.55 on HotpotQA. A3 is within one point of QLS only on 2Wiki and SQuAD. Thus neither rank fusion, one fixed structural rule, nor linear reweighting explains the six-dataset QLS result.

On the three regimes where fixed structure was most useful, A3 recovers 51.8% (WebQSP), 47.9% (HotpotQA), and 65.6% (MetaQA) of the selected-RRF to QLS R@5 gap. The remaining gap is consistent with nonlinear semantic/structural interaction; it is not evidence by itself that message passing is necessary, because QLS remains non-message-passing.

## Registered paired R@5 contrasts

| Dataset | A3 − RRF | A3 − seed-only | A3 − QLS | A3 − GNN | A3−QLS paired-query 95% CI | Holm p (A3−QLS) |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | +0.08 | +2.74 | +0.16 | -1.28 | [-0.79, +1.16] | 0.4394 |
| musique_clean | -0.73 | -11.56 | -11.77 | -12.73 | [-12.92, -10.67] | 4.511e-06 |
| webqsp | +12.01 | -7.05 | -11.16 | -10.89 | [-16.96, -5.78] | 7.837e-05 |
| hotpotqa_clean | +2.34 | +1.15 | -2.55 | -3.08 | [-2.97, -2.10] | 2.73e-05 |
| squad_clean | +0.19 | +0.19 | +0.27 | +0.17 | [-0.16, +0.69] | 0.01069 |
| metaqa | +10.74 | +1.24 | -5.63 | -5.65 | [-5.83, -5.44] | 3.295e-08 |

## A3 full effectiveness table

| Dataset | R@1 | R@5 | R@20 | MRR | FullCov@20 | LR |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 34.04 ± 0.32 | 68.57 ± 0.02 | 76.01 ± 0.12 | 87.03 ± 0.33 | 48.96 ± 0.21 | 0.05 |
| musique_clean | 29.97 ± 0.14 | 68.51 ± 0.09 | 81.93 ± 0.08 | 79.76 ± 0.20 | 61.73 ± 0.18 | 0.05 |
| webqsp | 3.79 ± 0.00 | 22.21 ± 0.00 | 35.93 ± 0.00 | 19.40 ± 0.00 | 27.67 ± 0.00 | 0.001 |
| hotpotqa_clean | 35.04 ± 0.12 | 74.58 ± 0.06 | 87.09 ± 0.08 | 81.51 ± 0.15 | 75.30 ± 0.17 | 0.05 |
| squad_clean | 59.37 ± 0.21 | 89.50 ± 0.03 | 96.74 ± 0.02 | 72.45 ± 0.16 | 96.74 ± 0.02 | 0.05 |
| metaqa | 9.76 ± 0.08 | 24.48 ± 0.03 | 31.38 ± 0.01 | 25.38 ± 0.11 | 24.57 ± 0.02 | 0.05 |

## Warm-cache systems accounting

| Dataset | A3 ms/query | A3 speedup vs QLS | A3 speedup vs GNN | A3 params | QLS params | GNN params | Total fixed cache GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 0.0502 | 2.32× | 8.59× | 19 | 213,506 | 213,568 | 0.142 |
| musique_clean | 0.0464 | 3.61× | 12.59× | 19 | 209,351 | 209,280 | 0.173 |
| webqsp | 0.0532 | 2.64× | 12.29× | 19 | 213,506 | 213,568 | 0.035 |
| hotpotqa_clean | 0.0535 | 3.17× | 9.07× | 19 | 213,506 | 213,440 | 0.904 |
| squad_clean | 0.0449 | 2.54× | 17.96× | 19 | 209,351 | 209,280 | 1.085 |
| metaqa | 0.0507 | 4.71× | 11.72× | 19 | 213,506 | 213,568 | 3.966 |

These are warm-cache post-retrieval measurements. They include cached-feature gathering and device transfer but exclude raw retrieval, graph/feature construction, and metric sorting. Structural and derived-cache build time/disk are preserved in the JSON analysis; no uncached real-world speedup is claimed.

## MetaQA R@5 by hop

| Hop | A3 linear | Seed-only | QLS-MLP | Seed-aware GNN |
|---:|---:|---:|---:|---:|
| 1 | 64.19 | 62.24 | 76.06 | 76.97 |
| 2 | 12.93 | 8.79 | 16.86 | 16.58 |
| 3 | 8.85 | 11.14 | 11.90 | 11.61 |

## Audit and stopping point

- All A1/A2/A3/confirmation test query indices and query-order hashes match exactly.
- All six A3 artifacts report five seeds, 19 parameters, validation-only selection, and exactly one label-based test evaluation per seed.
- The derived A3 cache contains only candidate IDs and Dense/SPLADE rank features; labels are built in memory and never persisted with features.
- The evaluator loads no graph adjacency, node embeddings, or query embeddings.
- Zero seed variance on WebQSP is reported rather than hidden; the convex linear scorer converged to the same ranking across shuffles.

**Package A is now closed.** Do not tune A3 against these tests. The evidence supports a graded capacity story—not a universal MLP win: fixed structure helps, linear weighting recovers part of that opportunity, nonlinear QLS is needed in four datasets, and seed-aware message passing retains a small-to-large R@5 lead except on SQuAD.
