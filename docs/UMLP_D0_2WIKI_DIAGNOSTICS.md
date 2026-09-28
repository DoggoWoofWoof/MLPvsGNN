# UMLP-D0: where u_mlp_v2_mix falls short on 2wiki

Analysis only, post hoc (configs/umlp_d0_2wiki_diagnostics.yaml, amendment A1). 2wiki dev, **V2_GATE half only** (6,290 queries); the held half was not read for any arm. No fit, no forward pass. Seed-mean arms over seeds 0-2 unless named `_s0`. Intervals: paired bootstrap, 1000 resamples, default_rng(0), 95 %.

`GAP_TO_MP` = twin minus gat_universal_v1 (what the failed twin cell measures); `GAP_TO_CORE` = twin minus qls_u_sota_v1 (the M3B non-MP incumbent on the 78-column contract -- the closest filed core, not an ablation); `GATING_EFFECT` = twin s0 minus the ungated u_mlp_v2 s0; `PROPAGATION_EFFECT` = gat_universal_v1 minus gat_no_mp_v1 (M3B's own MP contrast).

## Readings

| reading | verdict | evidence |
|---|---|---|
| R1 residual gating | **NOT_SUPPORTED** | median gates on 2wiki STRUCT_h1 0.395, STRUCT_h2 0.298, STRUCT_h3 0.390, FULL_h1 0.758, FULL_h2 0.328, FULL_h3 0.419; GAP_TO_CORE below zero in target cells: none; anywhere: none |
| R2 shortfall where propagation pays | **NOT_SUPPORTED** | distance 2+3 carry 0.0% of GAP_TO_MP (recall@5); PROPAGATION_EFFECT above zero in 2 and in 3: False -- a location, not an attribution |
| R3 set coverage | **NOT_SUPPORTED** | multi-gold (6,290 q): GAP_TO_MP full_coverage@5 -0.0784 vs hit@1 +0.0102 |

Slice concentration (the declared slices, as they fell on this population):

- gold_struct_distance: largest bucket `0 (seed is gold)` holds 6,267 of 6,290 queries (99.6%); empty: `3`
- gold_count: largest bucket `2` holds 4,946 of 6,290 queries (78.6%); empty: `1`
- gold_in_pool: largest bucket `all golds in pool` holds 5,700 of 6,290 queries (90.6%)
- hop: largest bucket `hop 0` holds 6,290 of 6,290 queries (100.0%)
- pool_size: largest bucket `<= 95` holds 2,213 of 6,290 queries (35.2%)
- depth_block_activation_STRUCT: largest bucket `a gold has h1 support` holds 4,937 of 6,290 queries (78.5%)
- depth_block_activation_FULL: largest bucket `a gold has h1 support` holds 5,444 of 6,290 queries (86.6%)

## Whole V2_GATE population

| arm | recall@5 | full_coverage@5 | hit@1 | mrr |
|---|---:|---:|---:|---:|
| twin | 0.8516 | 0.6721 | 0.8978 | 0.9389 |
| twin_ungated | 0.8502 | 0.6687 | 0.8903 | 0.9350 |
| core | 0.8383 | 0.6434 | 0.8658 | 0.9213 |
| mp | 0.8846 | 0.7505 | 0.8876 | 0.9311 |
| mp_struct | 0.8901 | 0.7552 | 0.9081 | 0.9441 |
| no_mp | 0.8404 | 0.6456 | 0.8950 | 0.9376 |
| gnn | 0.8996 | 0.7763 | 0.8908 | 0.9362 |
| fixed_rrf | 0.6123 | 0.2609 | 0.8762 | 0.9246 |
| twin_s0 | 0.8545 | 0.6752 | 0.9107 | 0.9463 |

| gap | recall@5 | full_coverage@5 | hit@1 | mrr |
|---|---|---|---|---|
| GAP_TO_MP | -0.0329 [-0.0363, -0.0297] | -0.0784 [-0.0852, -0.0719] | +0.0102 [+0.0043, +0.0156] | +0.0078 [+0.0044, +0.0109] |
| GAP_TO_CORE | +0.0134 [+0.0102, +0.0162] | +0.0287 [+0.0224, +0.0349] | +0.0321 [+0.0265, +0.0376] | +0.0176 [+0.0146, +0.0207] |
| GATING_EFFECT | +0.0043 [+0.0004, +0.0085] | +0.0065 [-0.0018, +0.0153] | +0.0203 [+0.0143, +0.0270] | +0.0113 [+0.0079, +0.0150] |
| PROPAGATION_EFFECT | +0.0442 [+0.0408, +0.0477] | +0.1049 [+0.0979, +0.1119] | -0.0074 [-0.0129, -0.0022] | -0.0065 [-0.0097, -0.0036] |
| GNN_V2_OVER_TWIN | +0.0479 [+0.0446, +0.0512] | +0.1042 [+0.0971, +0.1112] | -0.0070 [-0.0123, -0.0017] | -0.0027 [-0.0058, +0.0003] |

## Slice: gold_struct_distance

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| no gold in pool | 2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | -0.0% | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 0 (seed is gold) | 6,267 | 0.8541 | 0.8407 | 0.8869 | -0.0328 [-0.0363, -0.0295] | 99.2% | +0.0133 [+0.0103, +0.0162] | +0.0041 [+0.0002, +0.0082] | +0.0443 [+0.0409, +0.0474] | -0.0786 [-0.0857, -0.0713] |
| 1 | 8 | 0.3125 | 0.2292 | 0.3750 | -0.0625 [-0.3125, +0.1667] | 0.2% | +0.0833 [-0.0208, +0.2083] | +0.2500 [+0.0625, +0.4375] | +0.0417 [-0.1875, +0.2708] | -0.1250 [-0.3750, +0.0000] |
| 2 | 4 | 0.0000 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | -0.0% | +0.0000 [+0.0000, +0.0000] | -0.1250 [-0.3750, +0.0000] | -0.0417 [-0.1250, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 3 | 0 | - | - | - | - | -0.0% | - | - | - | - |
| >=4 or unreachable | 9 | 0.2222 | 0.2222 | 0.3426 | -0.1204 [-0.2778, +0.0280] | 0.5% | +0.0000 [-0.0926, +0.1481] | -0.0278 [-0.1667, +0.0833] | +0.0463 [-0.0556, +0.1299] | +0.0370 [+0.0000, +0.1111] |

## Slice: gold_count

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| 1 | 0 | - | - | - | - | -0.0% | - | - | - | - |
| 2 | 4,946 | 0.8749 | 0.8617 | 0.9042 | -0.0293 [-0.0331, -0.0254] | 69.9% | +0.0132 [+0.0098, +0.0168] | +0.0059 [+0.0010, +0.0104] | +0.0401 [+0.0363, +0.0437] | -0.0611 [-0.0684, -0.0543] |
| >=3 | 1,344 | 0.7661 | 0.7521 | 0.8125 | -0.0464 [-0.0528, -0.0401] | 30.1% | +0.0140 [+0.0086, +0.0193] | -0.0013 [-0.0087, +0.0063] | +0.0595 [+0.0531, +0.0660] | -0.1419 [-0.1587, -0.1252] |

## Slice: gold_in_pool

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| all golds in pool | 5,700 | 0.8849 | 0.8700 | 0.9190 | -0.0341 [-0.0375, -0.0304] | 93.8% | +0.0150 [+0.0119, +0.0184] | +0.0040 [-0.0004, +0.0082] | +0.0480 [+0.0443, +0.0516] | -0.0865 [-0.0940, -0.0789] |
| some | 588 | 0.5320 | 0.5339 | 0.5537 | -0.0217 [-0.0286, -0.0146] | 6.2% | -0.0018 [-0.0071, +0.0031] | +0.0077 [+0.0004, +0.0149] | +0.0072 [+0.0003, +0.0142] | +0.0000 [+0.0000, +0.0000] |
| none | 2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | -0.0% | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

## Slice: hop

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| hop 0 | 6,290 | 0.8516 | 0.8383 | 0.8846 | -0.0329 [-0.0363, -0.0297] | 100.0% | +0.0134 [+0.0102, +0.0162] | +0.0043 [+0.0004, +0.0085] | +0.0442 [+0.0408, +0.0477] | -0.0784 [-0.0852, -0.0719] |

## Slice: pool_size

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| <= 95 | 2,213 | 0.8578 | 0.8456 | 0.8933 | -0.0355 [-0.0417, -0.0292] | 38.0% | +0.0122 [+0.0072, +0.0174] | +0.0018 [-0.0056, +0.0088] | +0.0472 [+0.0415, +0.0531] | -0.0803 [-0.0919, -0.0676] |
| (95, 113] | 2,025 | 0.8544 | 0.8401 | 0.8823 | -0.0278 [-0.0335, -0.0219] | 27.2% | +0.0143 [+0.0093, +0.0190] | +0.0053 [-0.0015, +0.0122] | +0.0413 [+0.0353, +0.0468] | -0.0723 [-0.0835, -0.0606] |
| > 113 | 2,052 | 0.8423 | 0.8285 | 0.8774 | -0.0352 [-0.0412, -0.0297] | 34.8% | +0.0137 [+0.0084, +0.0191] | +0.0061 [-0.0005, +0.0130] | +0.0439 [+0.0386, +0.0495] | -0.0824 [-0.0947, -0.0702] |

## Slice: depth_block_activation_STRUCT

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| no gold in pool | 2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | -0.0% | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| a gold has h1 support | 4,937 | 0.8473 | 0.8304 | 0.8890 | -0.0417 [-0.0458, -0.0378] | 99.4% | +0.0169 [+0.0133, +0.0204] | +0.0033 [-0.0015, +0.0084] | +0.0572 [+0.0531, +0.0610] | -0.1006 [-0.1090, -0.0924] |
| else h2 | 603 | 0.8704 | 0.8733 | 0.8737 | -0.0033 [-0.0104, +0.0040] | 1.0% | -0.0029 [-0.0094, +0.0032] | +0.0054 [-0.0046, +0.0158] | +0.0004 [-0.0066, +0.0071] | -0.0077 [-0.0210, +0.0066] |
| else h3 | 457 | 0.8964 | 0.8895 | 0.8902 | +0.0062 [-0.0000, +0.0129] | -1.4% | +0.0069 [-0.0005, +0.0142] | +0.0098 [+0.0016, +0.0186] | -0.0069 [-0.0137, +0.0004] | +0.0160 [+0.0051, +0.0284] |
| no support at h<=3 | 291 | 0.8225 | 0.8242 | 0.8299 | -0.0074 [-0.0180, +0.0020] | 1.1% | -0.0017 [-0.0103, +0.0069] | +0.0112 [-0.0017, +0.0249] | -0.0046 [-0.0135, +0.0049] | +0.0034 [-0.0115, +0.0184] |

## Slice: depth_block_activation_FULL

| bucket | queries | twin R@5 | core R@5 | GAT R@5 | GAP_TO_MP R@5 | share of GAP_TO_MP | GAP_TO_CORE R@5 | GATING_EFFECT R@5 | PROPAGATION_EFFECT R@5 | GAP_TO_MP FC@5 |
|---|---:|---:|---:|---:|---|---:|---|---|---|---|
| no gold in pool | 2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | -0.0% | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| a gold has h1 support | 5,444 | 0.8458 | 0.8304 | 0.8840 | -0.0382 [-0.0421, -0.0345] | 100.4% | +0.0154 [+0.0121, +0.0186] | +0.0037 [-0.0013, +0.0080] | +0.0516 [+0.0477, +0.0555] | -0.0915 [-0.0994, -0.0836] |
| else h2 | 313 | 0.9044 | 0.9105 | 0.9028 | +0.0016 [-0.0064, +0.0096] | -0.2% | -0.0061 [-0.0157, +0.0019] | +0.0032 [-0.0096, +0.0160] | -0.0021 [-0.0085, +0.0043] | -0.0011 [-0.0170, +0.0139] |
| else h3 | 324 | 0.9162 | 0.9092 | 0.9108 | +0.0054 [-0.0018, +0.0139] | -0.8% | +0.0069 [-0.0015, +0.0170] | +0.0100 [+0.0015, +0.0185] | -0.0039 [-0.0106, +0.0028] | +0.0144 [+0.0010, +0.0298] |
| no support at h<=3 | 207 | 0.8321 | 0.8329 | 0.8394 | -0.0072 [-0.0189, +0.0036] | 0.7% | -0.0008 [-0.0081, +0.0081] | +0.0133 [-0.0012, +0.0290] | -0.0052 [-0.0157, +0.0044] | +0.0032 [-0.0113, +0.0177] |

## Block gates of u_mlp_v2_mix (V2_GATE, median of the per-query seed mean)

| block | 2wiki | metaqa | squad |
|---|---:|---:|---:|
| STRUCT_h1 | 0.395 | 0.498 | 0.451 |
| STRUCT_h2 | 0.298 | 0.610 | 0.278 |
| STRUCT_h3 | 0.390 | 0.514 | 0.342 |
| FULL_h1 | 0.758 | 0.393 | 0.657 |
| FULL_h2 | 0.328 | 0.601 | 0.285 |
| FULL_h3 | 0.419 | 0.422 | 0.351 |
| TYPED_h1 | 0.500 | 0.500 | 0.500 |
| TYPED_h2 | 0.492 | 0.578 | 0.493 |
| TYPED_h3 | 0.521 | 0.597 | 0.444 |

TYPED blocks are structurally inert on 2wiki (no typed STRUCT relation edge): their columns are zero whatever the gate.

Spearman correlation on 2wiki between a block's per-query gate and the query's gaps (recall@5):

| block | vs GAP_TO_CORE | vs GAP_TO_MP |
|---|---|---|
| STRUCT_h1 | -0.043 [-0.066, -0.021] | +0.067 [+0.045, +0.088] |
| STRUCT_h2 | -0.010 [-0.036, +0.015] | +0.041 [+0.014, +0.065] |
| STRUCT_h3 | +0.003 [-0.017, +0.025] | -0.058 [-0.079, -0.036] |
| FULL_h1 | +0.041 [+0.018, +0.064] | -0.048 [-0.071, -0.025] |
| FULL_h2 | -0.010 [-0.035, +0.015] | +0.032 [+0.005, +0.057] |
| FULL_h3 | +0.022 [+0.002, +0.041] | -0.037 [-0.059, -0.017] |

## Provenance

- masks: 6,290 V2_GATE queries recompiled by the frozen compiler in 375 s, 0 mismatches against the stored gold_dist_struct and fixed rrf arrays (sha256 `7f7768e6dc39fcfb`)
- inputs verified against the declaration's pins: 10 files
- configs/universal_v2.yaml unchanged (sha256 `bce0d1e92de4f184`)
- nothing here selects an architecture, a column or a threshold; a Universal-MLP-v2.1 declaration may cite it as motivation only
