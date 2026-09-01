# Edge-provenance results

All preregistered edge families are shown. Positive gaps mean learned message passing exceeds QLS-MLP on the same graph family.

| Dataset | Edge family | Directed edges | QLS R@5 | GNN R@5 | GNN − QLS |
|---|---|---:|---:|---:|---:|
| 2wiki_clean | sealed_a_multigraph | 855146 | 68.40 | 69.85 | +1.44 |
| 2wiki_clean | baseline_a_simple | 521614 | 67.81 | 70.05 | +2.25 |
| 2wiki_clean | symbolic_b | 1240748 | 68.13 | 69.64 | +1.51 |
| 2wiki_clean | knn_only | 269474 | 66.11 | 65.59 | -0.52 |
| 2wiki_clean | full_union_c | 1457378 | 67.16 | 69.44 | +2.28 |
| musique_clean | sealed_a_multigraph | 280108 | 80.28 | 81.24 | +0.96 |
| musique_clean | baseline_a_simple | 157898 | 80.18 | 81.25 | +1.07 |
| musique_clean | symbolic_b | 290086 | 80.98 | 81.53 | +0.55 |
| musique_clean | knn_only | 54812 | 80.23 | 81.00 | +0.77 |
| musique_clean | full_union_c | 327446 | 80.55 | 81.73 | +1.19 |
| webqsp | sealed_a_multigraph | 13379166 | 33.37 | 33.09 | -0.27 |
| webqsp | baseline_a_simple | 6621594 | 32.97 | 32.76 | -0.21 |
| webqsp | symbolic_b | 5228122 | 32.39 | 33.92 | +1.53 |
| webqsp | knn_only | 3345462 | 29.73 | 28.55 | -1.18 |
| webqsp | full_union_c | 8309152 | 33.05 | 32.64 | -0.42 |
| hotpotqa_clean | sealed_a_multigraph | 16223058 | 77.13 | 77.66 | +0.53 |
| hotpotqa_clean | baseline_a_simple | 9118344 | 76.73 | 77.23 | +0.51 |
| hotpotqa_clean | symbolic_b | 14971304 | 77.69 | 77.93 | +0.24 |
| hotpotqa_clean | knn_only | 2333304 | 73.54 | 73.33 | -0.21 |
| hotpotqa_clean | full_union_c | 16880560 | 76.46 | 77.05 | +0.59 |
| squad_clean | sealed_a_multigraph | 2857316 | 89.23 | 89.33 | +0.10 |
| squad_clean | baseline_a_simple | 1445712 | 89.29 | 89.36 | +0.07 |
| squad_clean | symbolic_b | 1642136 | 89.31 | 89.35 | +0.04 |
| squad_clean | knn_only | 56552 | 89.33 | 89.33 | +0.00 |
| squad_clean | full_union_c | 1682702 | 89.32 | 89.48 | +0.16 |
| metaqa | sealed_a_multigraph | 585728 | 30.11 | 30.13 | +0.02 |
| metaqa | baseline_a_simple | 329374 | 29.94 | 30.12 | +0.19 |
| metaqa | symbolic_b | 683970 | 30.44 | 30.30 | -0.14 |
| metaqa | knn_only | 110414 | 23.68 | 23.68 | +0.00 |
| metaqa | full_union_c | 775576 | 30.02 | 30.27 | +0.26 |

The sealed A multigraph is reused from the completed fairness confirmation. The simple-A row is the mandatory duplicate-normalization control; therefore a difference between sealed A and simple A cannot be misreported as edge semantics.
