# Equal-RRF candidate-budget results

All four preregistered budgets are reported for both matched models. No budget is selected using test effectiveness.

| Dataset | Budget | PoolCov | RRF R@5 | QLS R@5 | GNN R@5 | GNN − QLS | Candidates | Edges | Components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 50 | 74.80 | 68.18 | 67.35 | 67.51 | +0.16 | 50.0 | 73.2 | 32.0 |
| 2wiki_clean | 100 | 76.37 | 68.18 | 67.67 | 68.30 | +0.63 | 100.0 | 157.5 | 60.9 |
| 2wiki_clean | 200 | 77.93 | 68.18 | 68.19 | 69.20 | +1.02 | 200.0 | 350.7 | 112.8 |
| 2wiki_clean | 400 | 79.67 | 68.18 | 68.48 | 69.76 | +1.28 | 359.3 | 705.8 | 185.3 |
| musique_clean | 50 | 86.20 | 69.04 | 77.12 | 77.83 | +0.71 | 50.0 | 176.1 | 21.4 |
| musique_clean | 100 | 89.49 | 69.04 | 78.89 | 79.31 | +0.42 | 100.0 | 388.5 | 38.1 |
| musique_clean | 200 | 92.08 | 69.04 | 79.81 | 80.36 | +0.55 | 200.0 | 861.6 | 66.9 |
| musique_clean | 400 | 94.05 | 69.04 | 80.62 | 81.21 | +0.60 | 331.9 | 1558.1 | 96.6 |
| webqsp | 50 | 31.64 | 10.20 | 23.96 | 23.60 | -0.36 | 50.0 | 326.8 | 11.7 |
| webqsp | 100 | 37.72 | 10.20 | 26.75 | 25.70 | -1.05 | 100.0 | 686.0 | 22.9 |
| webqsp | 200 | 43.11 | 10.20 | 30.19 | 29.90 | -0.29 | 200.0 | 1363.5 | 47.0 |
| webqsp | 400 | 49.06 | 10.20 | 33.41 | 33.08 | -0.34 | 343.2 | 2323.7 | 78.9 |
| hotpotqa_clean | 50 | 87.67 | 72.18 | 76.62 | 76.83 | +0.21 | 50.0 | 239.7 | 24.3 |
| hotpotqa_clean | 100 | 90.04 | 72.18 | 76.91 | 77.23 | +0.31 | 100.0 | 951.7 | 47.6 |
| hotpotqa_clean | 200 | 91.60 | 72.18 | 76.94 | 77.38 | +0.44 | 200.0 | 1351.6 | 91.5 |
| hotpotqa_clean | 400 | 92.95 | 72.18 | 77.10 | 77.80 | +0.70 | 348.3 | 1997.0 | 152.3 |
| squad_clean | 50 | 98.04 | 89.14 | 89.20 | 89.12 | -0.08 | 50.0 | 698.4 | 15.1 |
| squad_clean | 100 | 98.86 | 89.14 | 89.20 | 89.29 | +0.09 | 100.0 | 1685.8 | 27.8 |
| squad_clean | 200 | 99.33 | 89.14 | 89.20 | 89.26 | +0.06 | 200.0 | 3743.6 | 50.8 |
| squad_clean | 400 | 99.49 | 89.14 | 89.27 | 89.41 | +0.14 | 318.6 | 6166.2 | 72.7 |
| metaqa | 50 | 21.40 | 10.22 | 20.43 | 20.53 | +0.09 | 50.0 | 85.5 | 28.8 |
| metaqa | 100 | 25.09 | 10.22 | 23.56 | 23.66 | +0.10 | 100.0 | 175.4 | 56.7 |
| metaqa | 200 | 29.26 | 10.22 | 26.87 | 26.98 | +0.11 | 200.0 | 370.6 | 109.4 |
| metaqa | 400 | 33.51 | 10.22 | 30.13 | 30.15 | +0.02 | 372.5 | 749.9 | 191.0 |

PoolCov is the mean fraction of a query's golds that reached the candidate pool. It is an oracle diagnostic, is never given to either model, and is not an achievable Recall@K: for a query with `g` golds and `p` of them pooled, Recall@K cannot exceed `min(p, K) / g`, which is below `p / g` whenever `g` exceeds `K`. The per-cut-off ceilings are reported in CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md. The systems fields in this table remain warm-cache measurements; the separate uncached unseen-embedding benchmark charges fusion, topology induction, QLS local summaries, transfer, forward, and top-K.
