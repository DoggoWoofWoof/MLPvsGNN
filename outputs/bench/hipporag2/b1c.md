# B1c: training-free walks on HippoRAG 2's settings

| setting | R0 (RRF) R@5 | chosen arm | its R@5 (all) | odd-half delta [95% CI] | call | HippoRAG 2 |
| --- | --- | --- | --- | --- | --- | --- |
| musique | 56.4 | R0 | 56.4 | +0.0 [+0.0, +0.0] | SAME | 74.7 |
| 2wiki | 71.2 | T5/structural | 71.7 | +0.3 [-0.2, +0.8] | SAME | 90.4 |
| hotpotqa | 85.0 | T5/structural | 85.7 | +0.1 [-0.3, +0.5] | SAME | 96.3 |

R@5 (all 1,000) of every arm:

| arm | musique | 2wiki | hotpotqa |
| --- | --- | --- | --- |
| R0 | 56.4 | 71.2 | 85.0 |
| T5/all | 56.4 | 71.2 | 85.0 |
| T5/structural | 55.9 | 71.7 | 85.7 |
| T5/ner | 56.4 | 71.2 | 85.0 |
| T5/knn | 56.4 | 71.2 | 85.0 |
| T5/ner+structural | 56.4 | 71.2 | 85.2 |
| RW/all | 28.8 | 47.4 | 64.8 |
| RW/structural | 17.5 | 55.2 | 62.0 |
| RW/ner | 24.3 | 32.8 | 43.0 |
| RW/knn | 28.8 | 36.0 | 49.5 |
| RW/ner+structural | 16.1 | 39.9 | 61.5 |

Ledger of the chosen arm's misses:

| setting | misses / golds | N1 | C1 | T1 | R10 | FAR | named | second of a chain | median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| musique | 1234 / 2648 | 0.190 | 0.423 | 0.150 | 0.024 | 0.212 | 0.190 | 0.859 | 35.5 |
| 2wiki | 787 / 2470 | 0.066 | 0.909 | 0.001 | 0.001 | 0.023 | 0.066 | 0.995 | 7 |
| hotpotqa | 287 / 2000 | 0.355 | 0.585 | 0.035 | 0.000 | 0.024 | 0.355 | 0.916 | 6 |

- musique: hops from top 5 {'1': 859, '2': 261, '3': 107, '-1': 7}; one hop by family {'structural': 0.315, 'ner': 0.554, 'knn': 0.327}; R@5 by kind {'2hop': 0.6525, '3hop1': 0.5254, '3hop2': 0.5388, '4hop1': 0.3333, '4hop2': 0.3611, '4hop3': 0.4355}
- 2wiki: hops from top 5 {'1': 751, '2': 13, '3': 13, '-1': 10}; one hop by family {'structural': 0.892, 'ner': 0.766, 'knn': 0.518}; R@5 by kind {'bridge_comparison': 0.5309, 'comparison': 0.9754, 'compositional': 0.6659, 'inference': 0.7315}
- hotpotqa: hops from top 5 {'1': 267, '3': 6, '2': 10, '-1': 4}; one hop by family {'structural': 0.906, 'ner': 0.62, 'knn': 0.648}; R@5 by kind {'bridge': 0.8298, 'comparison': 0.9709}
