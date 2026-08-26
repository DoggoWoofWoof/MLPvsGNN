# SA-MLP six-dataset fairness confirmation

Status: **all six datasets and five paired seeds complete**. SA-MLP is a non-message-passing fixed-structure model, not a topology-free model.

## Preregistered decision

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Fixed graph summaries add signal beyond the seed prior | 2/3 | 3/3 | supported |
| Seed prior explains at least 80% of the SA gain | 2/3 | 1/3 | not supported |
| Fixed summaries are non-inferior to seed-aware GNN within 1 R@5 point | 2/3 | 2/3 | supported |

The substitution gate conservatively requires both the paired-seed and paired-query 95% interval to clear the -1 point margin. All six datasets are reported.

| Dataset | SA - seed-only R@5 | Seed-prior recovery | SA - seed-aware GNN R@5 | Substitution |
|---|---:|---:|---:|---|
| metaqa | +6.86 | 44.6% | -0.02 | yes |
| webqsp | +4.11 | 42.0% | +0.27 | no |
| hotpotqa_clean | +3.70 | 81.2% | -0.53 | yes |

Across all six datasets, SA-MLP is 2.49–7.08× faster online and saves 90–2418 MiB of incremental peak GPU allocation; fixed-feature preprocessing and disk cache costs remain reported separately below.

**Stopping point:** the fairness-confirmation gate is closed. Freeze this as the primary result; do not tune these models or revisit test data. Any new mechanism, perturbation, or practical-width experiment requires a separate preregistered protocol. The universal MLP-over-GNN claim remains prohibited.

## 2wiki_clean

| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 29.10 ± 0.39 | 50.25 ± 0.50 | 63.32 ± 0.42 | 75.33 ± 0.61 | 31.40 ± 0.73 |
| seed_only | 29.00 ± 0.78 | 65.83 ± 0.47 | 73.03 ± 0.21 | 79.52 ± 1.09 | 44.19 ± 0.41 |
| sa_mlp | 29.23 ± 0.97 | 68.40 ± 0.42 | 77.75 ± 0.10 | 79.66 ± 1.27 | 52.04 ± 0.22 |
| seed_aware_gnn | 28.55 ± 0.44 | 69.85 ± 0.45 | 77.85 ± 0.10 | 78.61 ± 0.69 | 52.17 ± 0.15 |

| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| seed_only_minus_plain_mlp | recall@1 | -0.10 | [-0.93, +0.74] | [-1.32, +1.04] | 0.7641 |
| seed_only_minus_plain_mlp | recall@5 | +15.58 | [+15.21, +15.95] | [+14.41, +16.81] | 1.575e-07 |
| seed_only_minus_plain_mlp | recall@20 | +9.71 | [+9.10, +10.31] | [+8.79, +10.67] | 4.541e-06 |
| seed_only_minus_plain_mlp | mrr | +4.19 | [+3.15, +5.24] | [+2.40, +5.86] | 0.0011 |
| seed_only_minus_plain_mlp | full_coverage@20 | +12.79 | [+11.76, +13.81] | [+11.16, +14.41] | 1.25e-05 |
| sa_mlp_minus_seed_only | recall@1 | +0.22 | [-0.47, +0.92] | [-0.70, +1.16] | 1 |
| sa_mlp_minus_seed_only | recall@5 | +2.58 | [+2.09, +3.06] | [+1.81, +3.35] | 0.0004981 |
| sa_mlp_minus_seed_only | recall@20 | +4.72 | [+4.51, +4.93] | [+4.00, +5.43] | 1.585e-06 |
| sa_mlp_minus_seed_only | mrr | +0.14 | [-0.65, +0.93] | [-0.99, +1.28] | 1 |
| sa_mlp_minus_seed_only | full_coverage@20 | +7.85 | [+7.46, +8.25] | [+6.51, +9.23] | 2.563e-06 |
| sa_mlp_minus_seed_aware_gnn | recall@1 | +0.68 | [-0.68, +2.03] | [-0.49, +1.85] | 0.7705 |
| sa_mlp_minus_seed_aware_gnn | recall@5 | -1.44 | [-2.12, -0.77] | [-2.27, -0.65] | 0.02046 |
| sa_mlp_minus_seed_aware_gnn | recall@20 | -0.10 | [-0.24, +0.03] | [-0.31, +0.09] | 0.2023 |
| sa_mlp_minus_seed_aware_gnn | mrr | +1.05 | [-0.59, +2.70] | [-0.46, +2.57] | 0.3915 |
| sa_mlp_minus_seed_aware_gnn | full_coverage@20 | -0.13 | [-0.38, +0.11] | [-0.52, +0.25] | 0.4206 |

| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 204,928 | 21.2 | 0.0648 | 39.3 | n/a |
| seed_only | 213,574 | 25.7 | 0.2557 | 52.4 | 5058.3 |
| sa_mlp | 213,506 | 20.5 | 0.1165 | 53.0 | 5055.8 |
| seed_aware_gnn | 213,568 | 37.0 | 0.4316 | 143.3 | 5107.0 |

| Fixed-structure cost | Value |
|---|---:|
| SA latency / seed-aware GNN latency | 0.270 |
| Feature cache | 0.102 GiB |
| Feature precomputation | 9.3 s |

## musique_clean

| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 37.31 ± 0.11 | 76.72 ± 0.07 | 87.12 ± 0.19 | 89.09 ± 0.21 | 72.22 ± 0.26 |
| seed_only | 36.03 ± 0.49 | 80.08 ± 0.80 | 89.63 ± 0.17 | 88.24 ± 0.76 | 76.17 ± 0.42 |
| sa_mlp | 35.77 ± 0.81 | 80.28 ± 0.51 | 90.07 ± 0.31 | 87.80 ± 1.03 | 77.18 ± 0.50 |
| seed_aware_gnn | 36.54 ± 0.46 | 81.24 ± 0.33 | 90.23 ± 0.18 | 89.09 ± 0.55 | 77.19 ± 0.38 |

| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| seed_only_minus_plain_mlp | recall@1 | -1.27 | [-1.98, -0.57] | [-2.04, -0.57] | 0.02219 |
| seed_only_minus_plain_mlp | recall@5 | +3.35 | [+2.31, +4.40] | [+2.31, +4.33] | 0.00178 |
| seed_only_minus_plain_mlp | recall@20 | +2.52 | [+2.29, +2.74] | [+1.97, +3.07] | 1.296e-05 |
| seed_only_minus_plain_mlp | mrr | -0.86 | [-1.98, +0.27] | [-1.96, +0.20] | 0.2042 |
| seed_only_minus_plain_mlp | full_coverage@20 | +3.95 | [+3.47, +4.43] | [+2.86, +4.98] | 4.381e-05 |
| sa_mlp_minus_seed_only | recall@1 | -0.26 | [-1.17, +0.66] | [-1.10, +0.49] | 1 |
| sa_mlp_minus_seed_only | recall@5 | +0.20 | [-0.48, +0.88] | [-0.47, +0.93] | 0.9122 |
| sa_mlp_minus_seed_only | recall@20 | +0.44 | [+0.16, +0.72] | [+0.06, +0.79] | 0.01157 |
| sa_mlp_minus_seed_only | mrr | -0.43 | [-1.64, +0.77] | [-1.53, +0.52] | 1 |
| sa_mlp_minus_seed_only | full_coverage@20 | +1.01 | [+0.31, +1.72] | [+0.18, +1.83] | 0.02466 |
| sa_mlp_minus_seed_aware_gnn | recall@1 | -0.76 | [-2.15, +0.62] | [-1.70, +0.27] | 0.7705 |
| sa_mlp_minus_seed_aware_gnn | recall@5 | -0.96 | [-1.70, -0.23] | [-1.70, -0.30] | 0.08786 |
| sa_mlp_minus_seed_aware_gnn | recall@20 | -0.16 | [-0.61, +0.29] | [-0.56, +0.24] | 0.3747 |
| sa_mlp_minus_seed_aware_gnn | mrr | -1.29 | [-3.09, +0.51] | [-2.54, +0.08] | 0.3915 |
| sa_mlp_minus_seed_aware_gnn | full_coverage@20 | -0.01 | [-0.54, +0.52] | [-0.73, +0.69] | 0.9607 |

| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 204,928 | 24.9 | 0.0633 | 37.7 | n/a |
| seed_only | 209,137 | 50.9 | 0.4158 | 49.7 | 4963.9 |
| sa_mlp | 209,351 | 40.9 | 0.1675 | 50.0 | 4961.7 |
| seed_aware_gnn | 209,280 | 63.3 | 0.5844 | 301.2 | 5008.4 |

| Fixed-structure cost | Value |
|---|---:|
| SA latency / seed-aware GNN latency | 0.287 |
| Feature cache | 0.124 GiB |
| Feature precomputation | 12.7 s |

## webqsp

| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 13.71 ± 0.97 | 26.27 ± 1.07 | 37.57 ± 1.53 | 33.85 ± 1.44 | 28.43 ± 1.50 |
| seed_only | 11.91 ± 1.25 | 29.26 ± 1.44 | 38.60 ± 1.25 | 33.53 ± 0.32 | 29.69 ± 1.43 |
| sa_mlp | 17.38 ± 0.92 | 33.37 ± 1.14 | 41.39 ± 0.29 | 41.80 ± 1.15 | 32.33 ± 0.56 |
| seed_aware_gnn | 17.49 ± 1.67 | 33.09 ± 1.21 | 42.31 ± 0.41 | 42.55 ± 1.35 | 33.21 ± 0.53 |

| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| seed_only_minus_plain_mlp | recall@1 | -1.80 | [-4.08, +0.49] | [-5.84, +2.31] | 0.1886 |
| seed_only_minus_plain_mlp | recall@5 | +2.98 | [+0.22, +5.75] | [-0.82, +7.10] | 0.04019 |
| seed_only_minus_plain_mlp | recall@20 | +1.03 | [-0.23, +2.29] | [-1.49, +3.48] | 0.08675 |
| seed_only_minus_plain_mlp | mrr | -0.32 | [-2.40, +1.75] | [-4.24, +4.03] | 0.6874 |
| seed_only_minus_plain_mlp | full_coverage@20 | +1.26 | [+0.30, +2.21] | [-1.01, +3.77] | 0.02174 |
| sa_mlp_minus_seed_only | recall@1 | +5.46 | [+3.95, +6.98] | [+2.19, +8.83] | 0.002781 |
| sa_mlp_minus_seed_only | recall@5 | +4.11 | [+2.21, +6.01] | [+1.21, +7.51] | 0.0117 |
| sa_mlp_minus_seed_only | recall@20 | +2.79 | [+1.50, +4.08] | [+0.52, +5.45] | 0.007829 |
| sa_mlp_minus_seed_only | mrr | +8.27 | [+6.97, +9.57] | [+4.46, +11.98] | 0.0003008 |
| sa_mlp_minus_seed_only | full_coverage@20 | +2.64 | [+0.95, +4.33] | [+0.00, +5.79] | 0.02466 |
| sa_mlp_minus_seed_aware_gnn | recall@1 | -0.11 | [-1.35, +1.13] | [-3.37, +2.96] | 0.8162 |
| sa_mlp_minus_seed_aware_gnn | recall@5 | +0.27 | [-0.57, +1.12] | [-2.27, +2.77] | 1 |
| sa_mlp_minus_seed_aware_gnn | recall@20 | -0.92 | [-1.63, -0.21] | [-3.36, +1.13] | 0.06793 |
| sa_mlp_minus_seed_aware_gnn | mrr | -0.75 | [-1.72, +0.22] | [-3.96, +2.42] | 0.3915 |
| sa_mlp_minus_seed_aware_gnn | full_coverage@20 | -0.88 | [-2.06, +0.30] | [-3.52, +1.26] | 0.3238 |

| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 204,928 | 1.3 | 0.0634 | 38.4 | n/a |
| seed_only | 213,574 | 2.0 | 0.2816 | 50.9 | 8962.1 |
| sa_mlp | 213,506 | 1.8 | 0.1401 | 51.5 | 8963.1 |
| seed_aware_gnn | 213,568 | 3.7 | 0.6511 | 298.8 | 9040.5 |

| Fixed-structure cost | Value |
|---|---:|
| SA latency / seed-aware GNN latency | 0.215 |
| Feature cache | 0.030 GiB |
| Feature precomputation | 11.4 s |

## hotpotqa_clean

| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 32.05 ± 0.21 | 57.48 ± 0.20 | 72.08 ± 0.16 | 72.91 ± 0.33 | 51.61 ± 0.35 |
| seed_only | 35.35 ± 0.29 | 73.43 ± 0.04 | 84.02 ± 0.05 | 81.08 ± 0.38 | 69.57 ± 0.11 |
| sa_mlp | 36.49 ± 0.24 | 77.13 ± 0.20 | 88.56 ± 0.05 | 82.40 ± 0.27 | 78.49 ± 0.07 |
| seed_aware_gnn | 35.84 ± 0.24 | 77.66 ± 0.17 | 88.95 ± 0.04 | 81.65 ± 0.33 | 79.14 ± 0.09 |

| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| seed_only_minus_plain_mlp | recall@1 | +3.30 | [+2.85, +3.75] | [+2.88, +3.74] | 0.0001696 |
| seed_only_minus_plain_mlp | recall@5 | +15.95 | [+15.66, +16.23] | [+15.42, +16.51] | 6.15e-08 |
| seed_only_minus_plain_mlp | recall@20 | +11.94 | [+11.76, +12.13] | [+11.50, +12.41] | 3.54e-08 |
| seed_only_minus_plain_mlp | mrr | +8.17 | [+7.55, +8.79] | [+7.58, +8.81] | 1.671e-05 |
| seed_only_minus_plain_mlp | full_coverage@20 | +17.96 | [+17.46, +18.47] | [+17.16, +18.73] | 3.908e-07 |
| sa_mlp_minus_seed_only | recall@1 | +1.14 | [+0.69, +1.59] | [+0.71, +1.56] | 0.008802 |
| sa_mlp_minus_seed_only | recall@5 | +3.70 | [+3.43, +3.97] | [+3.33, +4.07] | 1.485e-05 |
| sa_mlp_minus_seed_only | recall@20 | +4.54 | [+4.46, +4.62] | [+4.24, +4.84] | 6.258e-08 |
| sa_mlp_minus_seed_only | mrr | +1.32 | [+0.77, +1.87] | [+0.81, +1.84] | 0.01036 |
| sa_mlp_minus_seed_only | full_coverage@20 | +8.91 | [+8.81, +9.01] | [+8.37, +9.46] | 9.419e-09 |
| sa_mlp_minus_seed_aware_gnn | recall@1 | +0.65 | [+0.31, +1.00] | [+0.30, +1.03] | 0.03234 |
| sa_mlp_minus_seed_aware_gnn | recall@5 | -0.53 | [-0.67, -0.39] | [-0.81, -0.24] | 0.002654 |
| sa_mlp_minus_seed_aware_gnn | recall@20 | -0.39 | [-0.49, -0.30] | [-0.54, -0.24] | 0.001711 |
| sa_mlp_minus_seed_aware_gnn | mrr | +0.75 | [+0.28, +1.22] | [+0.29, +1.23] | 0.05807 |
| sa_mlp_minus_seed_aware_gnn | full_coverage@20 | -0.66 | [-0.82, -0.50] | [-0.94, -0.37] | 0.001922 |

| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 204,928 | 126.8 | 0.0641 | 39.4 | n/a |
| seed_only | 213,313 | 180.4 | 0.2709 | 52.5 | 9301.6 |
| sa_mlp | 213,506 | 160.5 | 0.1693 | 53.1 | 9289.5 |
| seed_aware_gnn | 213,440 | 488.2 | 0.4854 | 2400.7 | 10805.0 |

| Fixed-structure cost | Value |
|---|---:|
| SA latency / seed-aware GNN latency | 0.352 |
| Feature cache | 0.650 GiB |
| Feature precomputation | 16.4 s |

## squad_clean

| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 50.00 ± 0.36 | 74.02 ± 0.23 | 86.41 ± 0.18 | 60.92 ± 0.26 | 86.41 ± 0.18 |
| seed_only | 54.08 ± 0.77 | 89.31 ± 0.28 | 95.54 ± 0.09 | 68.49 ± 0.55 | 95.54 ± 0.09 |
| sa_mlp | 54.35 ± 1.20 | 89.23 ± 0.11 | 95.78 ± 0.04 | 68.67 ± 0.77 | 95.78 ± 0.04 |
| seed_aware_gnn | 53.47 ± 0.80 | 89.33 ± 0.27 | 95.55 ± 0.08 | 68.15 ± 0.51 | 95.55 ± 0.08 |

| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| seed_only_minus_plain_mlp | recall@1 | +4.08 | [+2.97, +5.19] | [+3.17, +4.98] | 0.00209 |
| seed_only_minus_plain_mlp | recall@5 | +15.29 | [+14.91, +15.67] | [+14.68, +15.93] | 1.595e-07 |
| seed_only_minus_plain_mlp | recall@20 | +9.13 | [+8.87, +9.39] | [+8.63, +9.63] | 3.377e-07 |
| seed_only_minus_plain_mlp | mrr | +7.57 | [+6.81, +8.34] | [+6.93, +8.23] | 4.183e-05 |
| seed_only_minus_plain_mlp | full_coverage@20 | +9.13 | [+8.87, +9.39] | [+8.63, +9.63] | 3.908e-07 |
| sa_mlp_minus_seed_only | recall@1 | +0.28 | [-1.02, +1.57] | [-0.74, +1.19] | 1 |
| sa_mlp_minus_seed_only | recall@5 | -0.08 | [-0.56, +0.41] | [-0.45, +0.29] | 0.9122 |
| sa_mlp_minus_seed_only | recall@20 | +0.24 | [+0.15, +0.34] | [+0.09, +0.39] | 0.006539 |
| sa_mlp_minus_seed_only | mrr | +0.18 | [-0.67, +1.03] | [-0.47, +0.77] | 1 |
| sa_mlp_minus_seed_only | full_coverage@20 | +0.24 | [+0.15, +0.34] | [+0.09, +0.39] | 0.006539 |
| sa_mlp_minus_seed_aware_gnn | recall@1 | +0.88 | [-0.68, +2.44] | [-0.11, +2.06] | 0.7705 |
| sa_mlp_minus_seed_aware_gnn | recall@5 | -0.10 | [-0.54, +0.33] | [-0.49, +0.26] | 1 |
| sa_mlp_minus_seed_aware_gnn | recall@20 | +0.23 | [+0.14, +0.32] | [+0.08, +0.38] | 0.008966 |
| sa_mlp_minus_seed_aware_gnn | mrr | +0.52 | [-0.41, +1.45] | [-0.07, +1.23] | 0.3915 |
| sa_mlp_minus_seed_aware_gnn | full_coverage@20 | +0.23 | [+0.14, +0.32] | [+0.08, +0.38] | 0.008966 |

| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 204,928 | 151.9 | 0.0625 | 37.0 | n/a |
| seed_only | 209,137 | 224.5 | 0.2655 | 49.3 | 12927.8 |
| sa_mlp | 209,351 | 174.5 | 0.1136 | 49.3 | 12919.4 |
| seed_aware_gnn | 209,280 | 434.0 | 0.8049 | 2467.5 | 12980.7 |

| Fixed-structure cost | Value |
|---|---:|
| SA latency / seed-aware GNN latency | 0.141 |
| Feature cache | 0.776 GiB |
| Feature precomputation | 16.6 s |

## metaqa

| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 9.34 ± 0.11 | 17.73 ± 0.09 | 26.12 ± 0.07 | 24.97 ± 0.02 | 18.64 ± 0.07 |
| seed_only | 13.11 ± 0.12 | 23.25 ± 0.06 | 29.08 ± 0.07 | 30.71 ± 0.17 | 21.79 ± 0.07 |
| sa_mlp | 20.15 ± 0.07 | 30.11 ± 0.05 | 32.74 ± 0.01 | 40.73 ± 0.09 | 25.30 ± 0.01 |
| seed_aware_gnn | 20.96 ± 0.10 | 30.13 ± 0.02 | 32.63 ± 0.01 | 41.04 ± 0.06 | 25.25 ± 0.01 |

| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| seed_only_minus_plain_mlp | recall@1 | +3.78 | [+3.55, +4.01] | [+3.54, +4.02] | 8.569e-06 |
| seed_only_minus_plain_mlp | recall@5 | +5.52 | [+5.36, +5.68] | [+5.29, +5.77] | 1.934e-07 |
| seed_only_minus_plain_mlp | recall@20 | +2.95 | [+2.82, +3.09] | [+2.78, +3.14] | 1.703e-06 |
| seed_only_minus_plain_mlp | mrr | +5.74 | [+5.54, +5.94] | [+5.49, +6.00] | 9.517e-07 |
| seed_only_minus_plain_mlp | full_coverage@20 | +3.15 | [+3.01, +3.29] | [+2.95, +3.34] | 1.442e-06 |
| sa_mlp_minus_seed_only | recall@1 | +7.03 | [+6.85, +7.22] | [+6.76, +7.31] | 3.02e-07 |
| sa_mlp_minus_seed_only | recall@5 | +6.86 | [+6.77, +6.96] | [+6.64, +7.08] | 2.175e-08 |
| sa_mlp_minus_seed_only | recall@20 | +3.66 | [+3.58, +3.74] | [+3.50, +3.83] | 1.007e-07 |
| sa_mlp_minus_seed_only | mrr | +10.02 | [+9.81, +10.23] | [+9.71, +10.34] | 1.271e-07 |
| sa_mlp_minus_seed_only | full_coverage@20 | +3.50 | [+3.43, +3.58] | [+3.33, +3.69] | 9.52e-08 |
| sa_mlp_minus_seed_aware_gnn | recall@1 | -0.81 | [-0.87, -0.75] | [-0.98, -0.64] | 2.026e-05 |
| sa_mlp_minus_seed_aware_gnn | recall@5 | -0.02 | [-0.10, +0.06] | [-0.11, +0.07] | 1 |
| sa_mlp_minus_seed_aware_gnn | recall@20 | +0.11 | [+0.09, +0.13] | [+0.07, +0.15] | 0.00123 |
| sa_mlp_minus_seed_aware_gnn | mrr | -0.31 | [-0.38, -0.24] | [-0.48, -0.15] | 0.001381 |
| sa_mlp_minus_seed_aware_gnn | full_coverage@20 | +0.05 | [+0.03, +0.06] | [+0.00, +0.09] | 0.006533 |

| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |
|---|---:|---:|---:|---:|---:|
| plain_mlp | 204,928 | 279.3 | 0.0623 | 41.3 | n/a |
| seed_only | 213,574 | 556.3 | 0.3561 | 55.3 | 12607.3 |
| sa_mlp | 213,506 | 492.8 | 0.2386 | 55.8 | 12553.2 |
| seed_aware_gnn | 213,568 | 854.0 | 0.5945 | 205.2 | 14998.5 |

| Fixed-structure cost | Value |
|---|---:|
| SA latency / seed-aware GNN latency | 0.401 |
| Feature cache | 2.835 GiB |
| Feature precomputation | 20.5 s |

### MetaQA hop breakdown

| Hop | Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | plain_mlp | 21.05 ± 0.26 | 41.16 ± 0.24 | 60.52 ± 0.34 | 39.86 ± 0.16 | 51.51 ± 0.35 |
| 1 | seed_only | 34.96 ± 0.30 | 62.24 ± 0.22 | 71.97 ± 0.21 | 59.42 ± 0.28 | 64.03 ± 0.22 |
| 1 | sa_mlp | 50.61 ± 0.14 | 76.06 ± 0.07 | 79.74 ± 0.02 | 74.28 ± 0.08 | 73.35 ± 0.02 |
| 1 | seed_aware_gnn | 54.64 ± 0.22 | 76.97 ± 0.06 | 79.80 ± 0.02 | 77.44 ± 0.21 | 73.35 ± 0.03 |
| 2 | plain_mlp | 4.43 ± 0.07 | 8.90 ± 0.06 | 14.20 ± 0.06 | 12.06 ± 0.05 | 9.10 ± 0.05 |
| 2 | seed_only | 4.47 ± 0.05 | 8.79 ± 0.07 | 14.04 ± 0.06 | 12.35 ± 0.11 | 8.92 ± 0.09 |
| 2 | sa_mlp | 11.64 ± 0.11 | 16.86 ± 0.04 | 18.16 ± 0.01 | 26.34 ± 0.15 | 11.78 ± 0.01 |
| 2 | seed_aware_gnn | 11.06 ± 0.13 | 16.58 ± 0.03 | 17.99 ± 0.02 | 25.52 ± 0.05 | 11.71 ± 0.02 |
| 3 | plain_mlp | 6.28 ± 0.08 | 10.59 ± 0.08 | 14.58 ± 0.02 | 28.05 ± 0.13 | 5.69 ± 0.04 |
| 3 | seed_only | 6.89 ± 0.09 | 11.14 ± 0.03 | 14.86 ± 0.02 | 29.83 ± 0.25 | 5.77 ± 0.01 |
| 3 | sa_mlp | 7.79 ± 0.02 | 11.90 ± 0.05 | 15.19 ± 0.02 | 32.35 ± 0.11 | 5.90 ± 0.01 |
| 3 | seed_aware_gnn | 7.80 ± 0.04 | 11.61 ± 0.02 | 15.02 ± 0.03 | 31.86 ± 0.05 | 5.84 ± 0.02 |

## Interpretation contract

`seed-only - plain` measures the retrieval prior; `SA - seed-only` measures fixed graph computation; `SA - seed-aware GNN` compares fixed structural summaries with learned message passing. No contrast is collapsed into another.
