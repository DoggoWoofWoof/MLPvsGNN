# Deployable 2wiki compiled-kernel model (configs/deploy_ck_2wiki.yaml)

**Reading: `CK_PARTIAL`** for the primary kernel `ck_qi` on V2_HELD_CONFIRMATION recall@5 (co-read full_coverage@5); flags: `EDGES_CARRY`, `COMPILED_COST`.

The per-query reference `ck_full` (descriptive, the same bands): `CK_KEEPS` on V2_HELD_CONFIRMATION recall@5.

Every arm was fitted from the gold labels on 2wiki's own fit carve under universal-v2's frozen rule, early-stopped on its select carve, and scored on the M3B eval population, all on host_gpu_det. No GNN output, probe or kernel of the MP-Approx ladder enters any arm. The kernel is one hop of learned propagation (an MP arm by configs/universal_v2.yaml's definition), not QLS-U.

## Arms (seed means per query, then the mean)

### V2_HELD_CONFIRMATION (6286 queries)

| arm | parameters | recall@5 | full_coverage@5 | hit@1 | per-seed R@5 |
|---|---:|---:|---:|---:|---|
| twin | 330,955 | 0.8533 | 0.6753 | 0.8729 | 0.8525, 0.8465, 0.8609 |
| ck_qi | 377,803 | 0.8574 | 0.6850 | 0.8773 | 0.8546, 0.8543, 0.8633 |
| ck_full | 390,475 | 0.8926 | 0.7684 | 0.8759 | 0.8910, 0.8935, 0.8933 |
| ck_self | 343,883 | 0.8504 | 0.6691 | 0.8571 | 0.8485, 0.8501, 0.8527 |
| gnn | 420,932 | 0.8972 | 0.7820 | 0.8699 | 0.8992, 0.8908, 0.9016 |

| contrast | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| gnn_gain (gnn - twin) | +0.0439 [+0.0404, +0.0473] | +0.1067 [+0.0995, +0.1142] | -0.0030 [-0.0088, +0.0028] |
| ck_gain (ck_qi - twin) | +0.0041 [+0.0019, +0.0064] | +0.0097 [+0.0050, +0.0142] | +0.0044 [-0.0002, +0.0089] |
| edge_gain (ck_qi - ck_self) | +0.0070 [+0.0048, +0.0091] | +0.0159 [+0.0116, +0.0207] | +0.0202 [+0.0154, +0.0246] |
| self_gain (ck_self - twin) | -0.0029 [-0.0048, -0.0009] | -0.0063 [-0.0102, -0.0020] | -0.0157 [-0.0199, -0.0113] |
| ck_vs_gnn (ck_qi - gnn) | -0.0398 [-0.0430, -0.0367] | -0.0970 [-0.1039, -0.0905] | +0.0074 [+0.0016, +0.0130] |
| compiled_cost (ck_qi - ck_full) | -0.0352 [-0.0381, -0.0321] | -0.0834 [-0.0897, -0.0770] | +0.0014 [-0.0042, +0.0071] |
| ref_gain (ck_full - twin) | +0.0393 [+0.0362, +0.0424] | +0.0931 [+0.0864, +0.0996] | +0.0030 [-0.0030, +0.0084] |
| ref_vs_gnn (ck_full - gnn) | -0.0046 [-0.0068, -0.0024] | -0.0136 [-0.0179, -0.0090] | +0.0059 [+0.0010, +0.0111] |
| ck_share | 0.093 [0.045, 0.142] | 0.090 [0.050, 0.129] | -1.482 (not read: the denominator's interval reaches 0) |
| edge_share | 0.159 [0.111, 0.209] | 0.149 [0.108, 0.194] | -6.786 (not read: the denominator's interval reaches 0) |
| ref_share | 0.895 [0.851, 0.944] | 0.872 [0.835, 0.914] | -1.000 (not read: the denominator's interval reaches 0) |

Bands: recall@5 `CK_PARTIAL`, full_coverage@5 `CK_PARTIAL`, hit@1 `GNN_GAIN_ABSENT`. Reference: recall@5 `CK_KEEPS`, full_coverage@5 `CK_KEEPS`, hit@1 `GNN_GAIN_ABSENT`.

### V2_GATE (6290 queries)

| arm | parameters | recall@5 | full_coverage@5 | hit@1 | per-seed R@5 |
|---|---:|---:|---:|---:|---|
| twin | 330,955 | 0.8553 | 0.6775 | 0.8819 | 0.8553, 0.8485, 0.8620 |
| ck_qi | 377,803 | 0.8594 | 0.6859 | 0.8806 | 0.8609, 0.8552, 0.8620 |
| ck_full | 390,475 | 0.8938 | 0.7672 | 0.8855 | 0.8939, 0.8928, 0.8949 |
| ck_self | 343,883 | 0.8509 | 0.6674 | 0.8613 | 0.8502, 0.8517, 0.8508 |
| gnn | 420,932 | 0.9010 | 0.7835 | 0.8774 | 0.8994, 0.8957, 0.9080 |

| contrast | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| gnn_gain (gnn - twin) | +0.0457 [+0.0424, +0.0493] | +0.1059 [+0.0988, +0.1136] | -0.0045 [-0.0099, +0.0013] |
| ck_gain (ck_qi - twin) | +0.0041 [+0.0019, +0.0065] | +0.0083 [+0.0034, +0.0133] | -0.0013 [-0.0059, +0.0031] |
| edge_gain (ck_qi - ck_self) | +0.0084 [+0.0063, +0.0106] | +0.0184 [+0.0137, +0.0234] | +0.0193 [+0.0148, +0.0241] |
| self_gain (ck_self - twin) | -0.0044 [-0.0064, -0.0022] | -0.0101 [-0.0143, -0.0056] | -0.0206 [-0.0251, -0.0163] |
| ck_vs_gnn (ck_qi - gnn) | -0.0417 [-0.0448, -0.0384] | -0.0976 [-0.1046, -0.0904] | +0.0032 [-0.0029, +0.0090] |
| compiled_cost (ck_qi - ck_full) | -0.0345 [-0.0372, -0.0315] | -0.0813 [-0.0872, -0.0750] | -0.0049 [-0.0105, +0.0003] |
| ref_gain (ck_full - twin) | +0.0386 [+0.0352, +0.0417] | +0.0897 [+0.0828, +0.0969] | +0.0037 [-0.0017, +0.0096] |
| ref_vs_gnn (ck_full - gnn) | -0.0072 [-0.0094, -0.0051] | -0.0163 [-0.0206, -0.0121] | +0.0081 [+0.0035, +0.0131] |
| ck_share | 0.089 [0.042, 0.137] | 0.079 [0.034, 0.121] | 0.286 (not read: the denominator's interval reaches 0) |
| edge_share | 0.184 [0.136, 0.235] | 0.174 [0.129, 0.221] | -4.345 (not read: the denominator's interval reaches 0) |
| ref_share | 0.843 [0.796, 0.885] | 0.846 [0.807, 0.884] | -0.821 (not read: the denominator's interval reaches 0) |

Bands: recall@5 `CK_PARTIAL`, full_coverage@5 `CK_PARTIAL`, hit@1 `GNN_GAIN_ABSENT`. Reference: recall@5 `CK_KEEPS`, full_coverage@5 `CK_KEEPS`, hit@1 `GNN_GAIN_ABSENT`.

### whole (12576 queries)

| arm | parameters | recall@5 | full_coverage@5 | hit@1 | per-seed R@5 |
|---|---:|---:|---:|---:|---|
| twin | 330,955 | 0.8543 | 0.6764 | 0.8774 | 0.8539, 0.8475, 0.8615 |
| ck_qi | 377,803 | 0.8584 | 0.6854 | 0.8789 | 0.8577, 0.8547, 0.8627 |
| ck_full | 390,475 | 0.8932 | 0.7678 | 0.8807 | 0.8924, 0.8931, 0.8941 |
| ck_self | 343,883 | 0.8507 | 0.6682 | 0.8592 | 0.8494, 0.8509, 0.8517 |
| gnn | 420,932 | 0.8991 | 0.7827 | 0.8737 | 0.8993, 0.8932, 0.9048 |

| contrast | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| gnn_gain (gnn - twin) | +0.0448 [+0.0425, +0.0474] | +0.1063 [+0.1013, +0.1115] | -0.0037 [-0.0080, +0.0006] |
| ck_gain (ck_qi - twin) | +0.0041 [+0.0025, +0.0057] | +0.0090 [+0.0056, +0.0123] | +0.0016 [-0.0017, +0.0045] |
| edge_gain (ck_qi - ck_self) | +0.0077 [+0.0061, +0.0093] | +0.0172 [+0.0138, +0.0205] | +0.0197 [+0.0164, +0.0230] |
| self_gain (ck_self - twin) | -0.0036 [-0.0050, -0.0022] | -0.0082 [-0.0109, -0.0052] | -0.0182 [-0.0213, -0.0152] |
| ck_vs_gnn (ck_qi - gnn) | -0.0407 [-0.0430, -0.0383] | -0.0973 [-0.1021, -0.0922] | +0.0053 [+0.0010, +0.0093] |
| compiled_cost (ck_qi - ck_full) | -0.0348 [-0.0370, -0.0326] | -0.0824 [-0.0871, -0.0774] | -0.0017 [-0.0055, +0.0021] |
| ref_gain (ck_full - twin) | +0.0389 [+0.0366, +0.0411] | +0.0914 [+0.0865, +0.0964] | +0.0033 [-0.0007, +0.0073] |
| ref_vs_gnn (ck_full - gnn) | -0.0059 [-0.0075, -0.0042] | -0.0149 [-0.0180, -0.0118] | +0.0070 [+0.0034, +0.0106] |
| ck_share | 0.091 [0.057, 0.124] | 0.085 [0.054, 0.114] | -0.421 (not read: the denominator's interval reaches 0) |
| edge_share | 0.172 [0.136, 0.207] | 0.162 [0.131, 0.192] | -5.321 (not read: the denominator's interval reaches 0) |
| ref_share | 0.869 [0.835, 0.902] | 0.859 [0.832, 0.887] | -0.893 (not read: the denominator's interval reaches 0) |

Bands: recall@5 `CK_PARTIAL`, full_coverage@5 `CK_PARTIAL`, hit@1 `GNN_GAIN_ABSENT`. Reference: recall@5 `CK_KEEPS`, full_coverage@5 `CK_KEEPS`, hit@1 `GNN_GAIN_ABSENT`.

## Checks

- Compiled form of `ck_qi__H128__2wiki__s0`: largest score difference from the forward 7.63e-06; queries where a metric differs: 0.
- Compiled form of `ck_qi__H128__2wiki__s1`: largest score difference from the forward 7.63e-06; queries where a metric differs: 0.
- Compiled form of `ck_qi__H128__2wiki__s2`: largest score difference from the forward 3.81e-06; queries where a metric differs: 0.
- Compiled form of `ck_full__H128__2wiki__s0`: largest score difference from the forward 1.14e-05; queries where a metric differs: 0.
- Compiled form of `ck_full__H128__2wiki__s1`: largest score difference from the forward 1.53e-05; queries where a metric differs: 0.
- Compiled form of `ck_full__H128__2wiki__s2`: largest score difference from the forward 1.53e-05; queries where a metric differs: 0.
- Compiled form of `ck_self__H128__2wiki__s0`: largest score difference from the forward 0.00e+00; queries where a metric differs: 0.
- Compiled form of `ck_self__H128__2wiki__s1`: largest score difference from the forward 0.00e+00; queries where a metric differs: 0.
- Compiled form of `ck_self__H128__2wiki__s2`: largest score difference from the forward 0.00e+00; queries where a metric differs: 0.
- Compiled form of `ck_qi__H128__2wiki__s0__repeat`: largest score difference from the forward 7.63e-06; queries where a metric differs: 0.
- Repeat of `ck_qi__H128__2wiki__s0` in a fresh process: weights bit-identical True, metrics identical True.
- Eval population 12576 queries (ids 9473c881c281), halves 6290 / 6286; recall ceiling@5 0.9637; M3B fixed-rrf agreement True; MRR audit True.
- Inputs: the mirror's 2wiki record VERIFIED (sha256 3924f3a6e52b), freeze 58958f33a3af; family stores equal by content to the laptop's (built on the host: []).
- Batches (amendment 2): the first 16 batches of seeds [0, 1, 2] and the first 16 eval batches, packed on the host, equal the laptop's field by field (reference 17353b736b2f); each fit and the eval pass compared their own again before use: True.

## Fits

| fit | best epoch | epochs | select R@5 | minutes | weights |
|---|---:|---:|---:|---:|---|
| twin__H128__2wiki__s0 | 1 | 4 | 0.8917 | 13.8 | 7ca0d7d2ea0b |
| twin__H128__2wiki__s1 | 0 | 3 | 0.8885 | 10.5 | 645330806544 |
| twin__H128__2wiki__s2 | 0 | 3 | 0.8954 | 10.4 | 8d52a4e70770 |
| ck_qi__H128__2wiki__s0 | 1 | 4 | 0.8952 | 12.5 | 69557bb784da |
| ck_qi__H128__2wiki__s1 | 1 | 4 | 0.8935 | 12.5 | 43dd589ebdd3 |
| ck_qi__H128__2wiki__s2 | 0 | 3 | 0.8939 | 11.1 | f4b0d9ce6e8a |
| ck_full__H128__2wiki__s0 | 0 | 3 | 0.9138 | 11.1 | cbd7b8facc90 |
| ck_full__H128__2wiki__s1 | 2 | 5 | 0.9156 | 17.9 | 307ce52750eb |
| ck_full__H128__2wiki__s2 | 2 | 5 | 0.9195 | 17.9 | fa6452b946c0 |
| ck_self__H128__2wiki__s0 | 0 | 3 | 0.8909 | 10.6 | 268eb4df2da9 |
| ck_self__H128__2wiki__s1 | 0 | 3 | 0.8932 | 10.7 | b66888baf515 |
| ck_self__H128__2wiki__s2 | 0 | 3 | 0.8942 | 10.7 | 12e5d498808b |
| gnn__H128__2wiki__s0 | 0 | 3 | 0.9161 | 14.5 | ae2fa7d5e31f |
| gnn__H128__2wiki__s1 | 2 | 5 | 0.9136 | 22.7 | 13899d6a3765 |
| gnn__H128__2wiki__s2 | 0 | 3 | 0.9163 | 13.3 | 6feac3b84f76 |
| ck_qi__H128__2wiki__s0__repeat | 1 | 4 | 0.8952 | 12.0 | 69557bb784da |

The select numbers are each fit's early-stopping trace, never a result.

## Latency (systems only)

Batch of one on the first 500 eval queries, host_gpu_det (synchronised) and the host CPU at 8 threads, seed 0 of each arm. Compile is the unchanged per-query feature compile; the definitive end-to-end timing is the fast-compiler step's, not this file's.

| where | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| compile | 9.48 | 16.88 | 21.27 |
| pack | 1.98 | 7.91 | 11.26 |
| to_device | 3.45 | 6.18 | 7.79 |
| gpu:twin__H128__2wiki__s0 | 17.58 | 25.16 | 36.76 |
| cpu:twin__H128__2wiki__s0 | 5.38 | 7.26 | 8.37 |
| gpu:ck_qi__H128__2wiki__s0 | 22.13 | 37.42 | 48.09 |
| cpu:ck_qi__H128__2wiki__s0 | 7.52 | 10.26 | 11.36 |
| gpu:ck_full__H128__2wiki__s0 | 22.01 | 36.9 | 49.26 |
| cpu:ck_full__H128__2wiki__s0 | 7.27 | 9.86 | 11.06 |
| gpu:ck_self__H128__2wiki__s0 | 21.53 | 37.62 | 46.88 |
| cpu:ck_self__H128__2wiki__s0 | 6.52 | 8.65 | 9.69 |
| gpu:gnn__H128__2wiki__s0 | 36.8 | 56.9 | 77.93 |
| cpu:gnn__H128__2wiki__s0 | 12.94 | 16.64 | 18.7 |

## Mechanism (primary half, seed means)

| arm | delta ratio | top-1 changed |
|---|---:|---:|
| twin | 19.6343 | 0.4052 |
| ck_qi | 22.4334 | 0.3704 |
| ck_full | 26.5968 | 0.4025 |
| ck_self | 16.3489 | 0.4367 |
| gnn | 9.9230 | 0.5540 |

## Wording

every kernel arm is learned propagation, one hop, in-pool. No reading may be phrased as "message passing is unnecessary", "we do not need message passing" or "the MLP wins": the kernel is a message-passing arm, and whatever it keeps is kept by message passing. All numbers are dev-split numbers on one dataset, fitted and scored on host_gpu_det; none is a canonical result, and none is set beside a laptop or an earlier-stage number.

