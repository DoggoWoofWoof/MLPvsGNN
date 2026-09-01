# Uncached unseen-embedding systems results

Each request begins with a held-out query embedding and Dense/SPLADE ranked IDs. The timed path rebuilds equal-RRF candidates, retrieval seeds, candidate topology, and QLS local summaries before model inference.

| Dataset | Batch | Queries | QLS ms/query | GNN ms/query | QLS/GNN | QLS q/s | GNN q/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 1 | 1024 | 3.767 | 3.243 | 1.162 | 265.4 | 308.4 |
| 2wiki_clean | 16 | 1024 | 0.772 | 0.755 | 1.023 | 1294.8 | 1324.5 |
| musique_clean | 1 | 1024 | 3.919 | 3.136 | 1.250 | 255.2 | 318.9 |
| musique_clean | 16 | 1024 | 0.851 | 0.822 | 1.036 | 1174.5 | 1216.3 |
| webqsp | 1 | 159 | 3.489 | 2.916 | 1.196 | 286.6 | 342.9 |
| webqsp | 16 | 159 | 0.880 | 0.834 | 1.056 | 1136.2 | 1199.6 |
| hotpotqa_clean | 1 | 1024 | 3.998 | 2.886 | 1.385 | 250.2 | 346.5 |
| hotpotqa_clean | 16 | 1024 | 1.084 | 1.026 | 1.057 | 922.1 | 974.6 |
| squad_clean | 1 | 1024 | 4.391 | 3.840 | 1.143 | 227.8 | 260.4 |
| squad_clean | 16 | 1024 | 1.631 | 1.635 | 0.997 | 613.3 | 611.6 |
| metaqa | 1 | 1024 | 3.765 | 3.329 | 1.131 | 265.6 | 300.4 |
| metaqa | 16 | 1024 | 0.849 | 0.889 | 0.955 | 1177.3 | 1124.3 |

These are post-retrieval ranker timings, not raw-text end-to-end retrieval timings. Query encoding, Dense ANN lookup, and SPLADE index lookup are shared upstream and excluded for both methods. Cached operator-only numbers remain a separate reference.

## Cache break-even

Derived from the measured stage breakdown and the Package C warm-cache reference; it adds no timing. A per-query cache stores the products of fusion, seed construction, topology induction, and QLS summaries, so building one entry costs that prefix and reading it saves the difference between the uncached and cached paths. The break-even column is the number of *further* servings of the same query that repay building its entry.

This is compute-only: the frozen protocol measures static-asset bytes but not per-query cache footprint, so storage is excluded and these are lower bounds.

| Dataset | Batch | Model | Build ms | Uncached ms | Cached ms | Saved ms | Break-even repeats |
|---|---:|---|---:|---:|---:|---:|---:|
| 2wiki_clean | 1 | sa_mlp | 2.296 | 3.767 | 0.116 | 3.651 | 0.63 |
| 2wiki_clean | 1 | seed_aware_gnn | 0.815 | 3.243 | 0.457 | 2.786 | 0.29 |
| 2wiki_clean | 16 | sa_mlp | 0.618 | 0.772 | 0.116 | 0.656 | 0.94 |
| 2wiki_clean | 16 | seed_aware_gnn | 0.480 | 0.755 | 0.457 | 0.298 | 1.61 |
| musique_clean | 1 | sa_mlp | 2.438 | 3.919 | 0.113 | 3.806 | 0.64 |
| musique_clean | 1 | seed_aware_gnn | 0.890 | 3.136 | 0.474 | 2.662 | 0.33 |
| musique_clean | 16 | sa_mlp | 0.687 | 0.851 | 0.113 | 0.738 | 0.93 |
| musique_clean | 16 | seed_aware_gnn | 0.550 | 0.822 | 0.474 | 0.348 | 1.58 |
| webqsp | 1 | sa_mlp | 2.218 | 3.489 | 0.139 | 3.349 | 0.66 |
| webqsp | 1 | seed_aware_gnn | 0.802 | 2.916 | 0.628 | 2.288 | 0.35 |
| webqsp | 16 | sa_mlp | 0.727 | 0.880 | 0.139 | 0.741 | 0.98 |
| webqsp | 16 | seed_aware_gnn | 0.540 | 0.834 | 0.628 | 0.206 | 2.62 |
| hotpotqa_clean | 1 | sa_mlp | 2.580 | 3.998 | 0.130 | 3.867 | 0.67 |
| hotpotqa_clean | 1 | seed_aware_gnn | 1.099 | 2.886 | 0.446 | 2.440 | 0.45 |
| hotpotqa_clean | 16 | sa_mlp | 0.931 | 1.084 | 0.130 | 0.954 | 0.98 |
| hotpotqa_clean | 16 | seed_aware_gnn | 0.776 | 1.026 | 0.446 | 0.580 | 1.34 |
| squad_clean | 1 | sa_mlp | 2.984 | 4.391 | 0.114 | 4.277 | 0.70 |
| squad_clean | 1 | seed_aware_gnn | 1.632 | 3.840 | 0.953 | 2.888 | 0.57 |
| squad_clean | 16 | sa_mlp | 1.469 | 1.631 | 0.114 | 1.517 | 0.97 |
| squad_clean | 16 | seed_aware_gnn | 1.333 | 1.635 | 0.953 | 0.682 | 1.95 |
| metaqa | 1 | sa_mlp | 2.326 | 3.765 | 0.115 | 3.650 | 0.64 |
| metaqa | 1 | seed_aware_gnn | 0.932 | 3.329 | 0.459 | 2.870 | 0.32 |
| metaqa | 16 | sa_mlp | 0.696 | 0.849 | 0.115 | 0.734 | 0.95 |
| metaqa | 16 | seed_aware_gnn | 0.605 | 0.889 | 0.459 | 0.431 | 1.40 |
