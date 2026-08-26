# Structure-Aware MLP one-seed screening result

Status: **the preregistered gate passed on all three datasets**. Values below are seed 0 only and are screening evidence, not five-seed paper estimates.

## Primary gate

| Dataset | Frozen MLP R@5 | Frozen GNN R@5 | SA-MLP R@5 | SA-GNN delta | Gap closure | Pass |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 17.78 | 25.55 | 30.04 | 4.49 | 1.58x | yes |
| webqsp | 27.66 | 31.10 | 32.66 | 1.55 | 1.45x | yes |
| hotpotqa_clean | 57.44 | 62.04 | 77.22 | 15.18 | 4.30x | yes |

The frozen rule required at least 0.50x closure on two of three datasets. Observed closure is 1.58x on MetaQA, 1.45x on WebQSP, and 4.30x on HotpotQA.

## Ablation ladder

| Dataset | Frozen MLP | Interaction | Static | Query-local | Full SA | Frozen GNN |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 17.78 | 17.59 | 12.43 | 30.02 | 30.04 | 25.55 |
| webqsp | 27.66 | 25.34 | 18.05 | 28.29 | 32.66 | 31.10 |
| hotpotqa_clean | 57.44 | 57.43 | 20.43 | 73.22 | 77.22 | 62.04 |

Static graph descriptors alone are harmful in all three settings. Query-local distance/path/PPR descriptors account for essentially the entire MetaQA gain and most of the HotpotQA gain. WebQSP requires the combined interaction and query-local model; neither family closes its GNN gap alone.

## MetaQA diagnostic

| Hop | Queries | Frozen MLP R@5 | Frozen GNN R@5 | Query-local R@5 | SA-MLP R@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 9947 | 41.23 | 67.15 | 75.78 | 75.96 |
| 2 | 14872 | 8.99 | 11.28 | 16.83 | 16.82 |
| 3 | 14274 | 10.59 | 11.42 | 11.87 | 11.82 |

SA-MLP exceeds the frozen GNN at every hop. The improvement is largest at one and two hops, while the three-hop difference is small. This strengthens the result that native query hop count is not a monotonic proxy for message-passing value.

## Systems cost

| Dataset | Cache GiB | Preprocess s | SA ms/q | GNN ms/q | GNN/SA latency | SA incr. GPU MiB | GNN incr. GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 2.835 | 20.5 | 0.2017 | 0.2443 | 1.21x | 55.8 | 205.2 |
| webqsp | 0.030 | 11.4 | 0.1384 | 0.4119 | 2.98x | 51.5 | 298.5 |
| hotpotqa_clean | 0.650 | 16.4 | 0.1264 | 0.2333 | 1.85x | 53.1 | 2400.7 |

The learned SA forward pass is 1.21--2.98x faster than the selected GNN, not the 3.6--9.9x advantage of the plain MLP, because online cache lookup and the explicit scoring head add work. GPU allocation remains far below the GNN, especially on HotpotQA. The method also shifts cost to CPU/disk: caches range from 0.030 to 2.835 GiB and total process RSS is high because frozen arrays and memory maps coexist.

## Required fairness control before the paper claim

The query-local feature set includes a distance-0 bucket for the frozen dense/SPLADE retrieval seeds. This exposes seed membership explicitly, whereas the frozen GNN received candidate embeddings and adjacency but no seed indicator. Therefore this screen establishes that the registered fixed-feature package beats the old GNN; it does not yet isolate how much comes from graph paths/PPR versus the retrieval-seed prior. Confirmation must retain the frozen SA architecture and add a seed-only control (and, for the strongest causal comparison, a seed-aware GNN control).

No test-driven feature or architecture change was made. The combined SA-MLP is now eligible to be frozen for five-seed confirmation, but the one-seed values must not be presented as final estimates.
