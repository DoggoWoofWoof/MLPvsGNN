# ck_full end to end against the GNN on 2wiki (configs/deploy_ck_full_2wiki.yaml)

**Retention: `FULL_KEEPS`** on V2_HELD_CONFIRMATION recall@5; **speed: `E2E_FASTER`** on the end-to-end p50 ratio ck_full / gnn; flags: none.

ck_full keeps 0.895 [0.851, 0.944] of the GNN's recall@5 gain over the twin on V2_HELD_CONFIRMATION at 0.656 [0.644, 0.670] of the GNN's end-to-end p50 latency (0.624 at p95, 0.534 at p99) and 0.877 of its model-side memory increase; laptop CPU, 8 threads, a batch of one, the fast compiler wired in. The twin, the no-propagation floor: 0.664 of the GNN's p50 (0.604 at p95, 0.527 at p99).

ck_full is a compressed, query-conditioned one-hop message-passing approximation (its kernel is learned propagation). Every arm is deploy_ck_2wiki's sealed fit, unchanged; the retention was scored again on this laptop's CPU at 8 threads with the fast feature compiler wired in and compared with the reference compile on every query. Every number here is a laptop number on 2wiki's dev population; none is set beside a host number.

## Retention (seed means per query, then the mean)

### V2_HELD_CONFIRMATION (6286 queries)

| arm | parameters | recall@5 | full_coverage@5 | hit@1 | per-seed R@5 |
|---|---:|---:|---:|---:|---|
| twin | 330,955 | 0.8533 | 0.6753 | 0.8729 | 0.8525, 0.8465, 0.8609 |
| ck_full | 390,475 | 0.8926 | 0.7684 | 0.8759 | 0.8910, 0.8935, 0.8933 |
| gnn | 420,932 | 0.8972 | 0.7820 | 0.8699 | 0.8992, 0.8908, 0.9016 |

| contrast | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| gnn_gain (gnn - twin) | +0.0439 [+0.0404, +0.0473] | +0.1067 [+0.0995, +0.1142] | -0.0030 [-0.0088, +0.0028] |
| full_gain (ck_full - twin) | +0.0393 [+0.0362, +0.0424] | +0.0931 [+0.0864, +0.0996] | +0.0030 [-0.0030, +0.0084] |
| full_vs_gnn (ck_full - gnn) | -0.0046 [-0.0068, -0.0024] | -0.0136 [-0.0179, -0.0090] | +0.0059 [+0.0010, +0.0111] |
| full_share | 0.895 [0.851, 0.944] | 0.872 [0.835, 0.914] | -1.000 (not read: the denominator's interval reaches 0) |

Bands: recall@5 `FULL_KEEPS`, full_coverage@5 `FULL_KEEPS`, hit@1 `GNN_GAIN_ABSENT`.

### V2_GATE (6290 queries)

| arm | parameters | recall@5 | full_coverage@5 | hit@1 | per-seed R@5 |
|---|---:|---:|---:|---:|---|
| twin | 330,955 | 0.8553 | 0.6775 | 0.8819 | 0.8553, 0.8485, 0.8620 |
| ck_full | 390,475 | 0.8938 | 0.7672 | 0.8855 | 0.8939, 0.8928, 0.8949 |
| gnn | 420,932 | 0.9010 | 0.7835 | 0.8774 | 0.8994, 0.8957, 0.9080 |

| contrast | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| gnn_gain (gnn - twin) | +0.0457 [+0.0424, +0.0493] | +0.1059 [+0.0988, +0.1136] | -0.0045 [-0.0099, +0.0013] |
| full_gain (ck_full - twin) | +0.0386 [+0.0352, +0.0417] | +0.0897 [+0.0828, +0.0969] | +0.0037 [-0.0017, +0.0096] |
| full_vs_gnn (ck_full - gnn) | -0.0072 [-0.0094, -0.0051] | -0.0163 [-0.0206, -0.0121] | +0.0081 [+0.0035, +0.0131] |
| full_share | 0.843 [0.796, 0.885] | 0.846 [0.807, 0.884] | -0.821 (not read: the denominator's interval reaches 0) |

Bands: recall@5 `FULL_KEEPS`, full_coverage@5 `FULL_KEEPS`, hit@1 `GNN_GAIN_ABSENT`.

### whole (12576 queries)

| arm | parameters | recall@5 | full_coverage@5 | hit@1 | per-seed R@5 |
|---|---:|---:|---:|---:|---|
| twin | 330,955 | 0.8543 | 0.6764 | 0.8774 | 0.8539, 0.8475, 0.8615 |
| ck_full | 390,475 | 0.8932 | 0.7678 | 0.8807 | 0.8924, 0.8931, 0.8941 |
| gnn | 420,932 | 0.8991 | 0.7827 | 0.8737 | 0.8993, 0.8932, 0.9048 |

| contrast | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| gnn_gain (gnn - twin) | +0.0448 [+0.0425, +0.0474] | +0.1063 [+0.1013, +0.1115] | -0.0037 [-0.0080, +0.0006] |
| full_gain (ck_full - twin) | +0.0389 [+0.0366, +0.0411] | +0.0914 [+0.0865, +0.0964] | +0.0033 [-0.0007, +0.0073] |
| full_vs_gnn (ck_full - gnn) | -0.0059 [-0.0075, -0.0042] | -0.0149 [-0.0180, -0.0118] | +0.0070 [+0.0034, +0.0106] |
| full_share | 0.869 [0.835, 0.902] | 0.859 [0.832, 0.887] | -0.893 (not read: the denominator's interval reaches 0) |

Bands: recall@5 `FULL_KEEPS`, full_coverage@5 `FULL_KEEPS`, hit@1 `GNN_GAIN_ABSENT`.

## Latency (laptop CPU, 8 threads, a batch of one)

1024 held queries x 3 rounds per arm, each arm in a fresh process per round in a rotated order; percentiles of the pooled per-query totals, 95% cluster-bootstrap intervals (a drawn query brings its rounds).

| arm | e2e p50 ms | e2e p95 ms | e2e p99 ms | forward p50 ms | graph part p50 ms | reference-compile e2e p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| twin | 25.12 [24.64, 25.77] | 47.23 [45.74, 48.90] | 61.56 [57.10, 64.88] | 11.02 | 0.14 | 33.37 |
| ck_full | 24.81 [24.39, 25.39] | 48.80 [47.35, 50.02] | 62.44 [59.54, 64.81] | 12.07 | 3.60 | 32.27 |
| gnn | 37.81 [37.18, 38.39] | 78.23 [75.45, 80.85] | 116.91 [106.58, 138.13] | 25.29 | 14.61 | 50.00 |

ck_full's kernel module alone (the query-dependent kernel built and applied): p50 2.929 [2.885, 2.976] ms, p95 6.364, p99 9.461.

| ratio | p50 | p95 | p99 |
|---|---|---|---|
| e2e ck_full/gnn | 0.656 [0.644, 0.670] | 0.624 [0.602, 0.647] | 0.534 [0.451, 0.596] |
| e2e twin/gnn | 0.664 [0.652, 0.680] | 0.604 [0.581, 0.634] | 0.527 [0.443, 0.600] |
| e2e ck_full/twin | 0.988 [0.969, 1.004] | 1.033 [0.995, 1.064] | 1.014 [0.945, 1.092] |
| forward ck_full/gnn | 0.477 [0.466, 0.490] | 0.481 [0.463, 0.500] | 0.407 [0.352, 0.446] |
| forward twin/gnn | 0.436 [0.428, 0.447] | 0.373 [0.360, 0.387] | 0.327 [0.279, 0.356] |
| forward ck_full/twin | 1.095 [1.072, 1.113] | 1.289 [1.244, 1.339] | 1.245 [1.164, 1.367] |
| graph part ck_full/gnn | 0.247 [0.241, 0.252] | 0.272 [0.258, 0.287] | 0.294 [0.266, 0.318] |

Median per path stage (e2e_fast, ms):

| arm | pool | gather | compile | pack | forward | topk |
|---|---:|---:|---:|---:|---:|---:|
| twin | 1.065 | 1.461 | 6.239 | 3.755 | 11.022 | 0.069 |
| ck_full | 0.653 | 1.261 | 5.389 | 3.304 | 12.070 | 0.051 |
| gnn | 0.631 | 1.223 | 5.803 | 3.503 | 25.289 | 0.040 |

Other speed bands: p95/p99 {'p95': 'E2E_FASTER', 'p99': 'E2E_FASTER'}, forward alone `E2E_FASTER`.

## Memory

| arm | model-side increase MiB (median over rounds) | sampled increase MiB | peak moved (rounds) | parameter KiB |
|---|---:|---:|---:|---:|
| twin | 6.39 | 6.39 | 3 | 1292.8 |
| ck_full | 6.66 | 6.66 | 3 | 1525.3 |
| gnn | 7.59 | 7.59 | 3 | 1644.3 |

| process | startup s | ready working set MiB | peak working set MiB | idle at start |
|---|---:|---:|---:|---|
| twin__r0 | 8.2 | 1739 | 2432 | True |
| twin__r1 | 9.4 | 1739 | 2427 | True |
| twin__r2 | 9.0 | 1739 | 2427 | True |
| ck_full__r0 | 8.2 | 1740 | 2432 | True |
| ck_full__r1 | 8.2 | 1741 | 2434 | True |
| ck_full__r2 | 12.2 | 1741 | 2434 | True |
| gnn__r0 | 8.8 | 1744 | 2437 | True |
| gnn__r1 | 7.9 | 1744 | 2440 | True |
| gnn__r2 | 10.1 | 1744 | 2431 | True |

## Checks

- Fast compile: 12576 eval queries and 1088 sampled queries compiled by both compilers, bit-identical on every array of every query.
- Batches (amendment 2): {'batches_compared': 16, 'reference_batches': 16, 'equal': True, 'reference_sha256': '17353b736b2faf14add3d5148c66647fe96d120f242d7829fa41add7df821e2c'}.
- Pools rebuilt from the first-stage lists equal the prepared pools on all 1088 sampled and warm-up queries.
- Timed paths' top-5 against the cached pass: {'twin__r0': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'twin__r1': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'twin__r2': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'ck_full__r0': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'ck_full__r1': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'ck_full__r2': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'gnn__r0': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'gnn__r1': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}, 'gnn__r2': {'e2e_fast': 0, 'e2e_reference': 0, 'breakdown': 0}}.
- Eval population 12576 queries (ids 9473c881c281); recall ceiling@5 0.9637; M3B fixed-rrf agreement True; MRR audit True.
- Host agreement (systems count, not read): twin__H128__2wiki__s0 0, twin__H128__2wiki__s1 0, twin__H128__2wiki__s2 0, ck_full__H128__2wiki__s0 0, ck_full__H128__2wiki__s1 0, ck_full__H128__2wiki__s2 0, gnn__H128__2wiki__s0 0, gnn__H128__2wiki__s1 0, gnn__H128__2wiki__s2 0.
- Compiled form of `ck_full__H128__2wiki__s0` (an exactness check, never timed): largest score difference 1.14e-05, queries where a metric differs 0.
- Compiled form of `ck_full__H128__2wiki__s1` (an exactness check, never timed): largest score difference 1.91e-05, queries where a metric differs 0.
- Compiled form of `ck_full__H128__2wiki__s2` (an exactness check, never timed): largest score difference 1.91e-05, queries where a metric differs 0.
- Power scheme (read, never changed): Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced); driver deviations: none.

## Wording

ck_full is named a compressed, query-conditioned one-hop message-passing approximation. No reading may call it a non-message-passing model, an MLP or a QLS-U replacement, and none may be phrased as "message passing is unnecessary", "we do not need message passing" or "the MLP wins": whatever ck_full keeps is kept by message passing. No speed claim is made except the band this file reads. All numbers are dev-split numbers on one dataset, scored and timed on the laptop; none is a canonical result, and none is set beside a host number or an earlier stage's number.

