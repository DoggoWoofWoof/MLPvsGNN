# Universal-v2 post-pilot replication (amendment 4)

**ORIGINAL PILOT STATUS: PILOT_FAILED** (commit `d2850ab`, `run_record_2026_09_21`, `docs/UNIVERSAL_V2_PILOT.md` as filed -- not re-rendered, not reinterpreted).

**POST-PILOT REPLICATION STATUS: GNN_REPLICATION_ONLY** -- GNN_GATE family (`u_gnn_v2_ef`) **REPLICATION_PASS** / TWIN_GATE family (`u_mlp_v2_mix`) **REPLICATION_FAIL**.

The two statuses are separate: nothing here renames the original pilot's result, and a replication pass is never `PILOT_PASS` (`amendment_4_2026_09_21` status_vocabulary). GNN_REPLICATION_ONLY and TWIN_REPLICATION_ONLY replicate that family only, never the proposed universal pair. Read on V2_GATE once (2026-09-22T18:50:03Z, status line `RUN_PILOT_FAILED`, unchanged). Interpretation rule for this outcome (verbatim from the amendment): the proposed universal GNN is stable enough to continue; the universal non-MP twin is not yet sufficiently robust

## 1. What was frozen and verified

- The amendment `amendment_4_2026_09_21` was filed and committed before any fit it authorises; it authorises {'u_gnn_v2_ef': [1, 2], 'u_mlp_v2_mix': [1, 2]} and nothing else.
- The selected arms are those of `selection.json` and `gate_record.json` ({'gnn': 'u_gnn_v2_ef', 'twin': 'u_mlp_v2_mix'}); the pilot's sidecars hash to what `run_record_2026_09_21` pins: `selection.json` `3ed48ae3d043…`, `gate_record.json` `8dcc217b08a7…`, `held_record.json` `862acbadaf72…`.
- Thresholds from `amendment_1_2026_09_19.pilot_gate_amended`, unchanged; the aggregation is scripts/universal_v2_report.py seed_mean_cells (the pilot's seed confirmation, pilot_gate.on_pass): per cell the per-query mean over seeds 0-2 on V2_GATE against the frozen threshold; a paired cell by the declared bootstrap of that per-query mean; paired intervals: 1000 resamples, `default_rng(0)`, 95 % percentile.
- Every seed-1/2 fit refused to start unless its training rule, parameter count, columns, core sha256, evidence fields and relation bank equalled the seed-0 record of the same arm (the fit records carry `replication.checks_against_seed_0`); seed 0 was not retrained (its records exist and are not repeated).
- M3B pins verified at every stage; the M3B reference arrays: metaqa `a4739cf66a8d…`, 2wiki `d2196c2f7769…`, squad `c14fe820295d…`.
- Queries on V2_GATE: metaqa 19,738, 2wiki 6,290, squad 5,841.

## 2. The fits

| key | family | seed | parameters | best epoch | epochs run | select macro R@5 | hours | peak RSS GB | authorised by | state sha256 (checkpoint hash) | record sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `u_gnn_v2_ef__H128__s0` | gnn | 0 | 420,932 | 1 | 4 | 0.8777 | 5.02 | 8.64 | pilot (seed 0) | `9f749ad346afcef479f84436019771e0b940a1e647e4960908acf4105ea2ed0e` | `bfd3e8e5957b259b2d8b38f85008f659f48e00eb79cdf43220aeb39817993009` |
| `u_gnn_v2_ef__H128__s1` | gnn | 1 | 420,932 | 1 | 4 | 0.8796 | 7.12 | 6.42 | amendment_4_2026_09_21 | `9b3a5521babb814e83e8a0a40a94b9abd92bdee7f7f58b04f686558d0522c367` | `6e1efe41c8c2f27bfd1d37fa26e9b8e00685e1ffb8d12f8528d489b6b3a9a60a` |
| `u_gnn_v2_ef__H128__s2` | gnn | 2 | 420,932 | 1 | 4 | 0.8763 | 10.38 | 6.94 | amendment_4_2026_09_21 | `4575c41822f56d99b843aaa524a5eb407fa038ebc24a6ad04f4fef786433d4c4` | `4cb612e195c85fde11edaec8b6f0dcd293fc706b2554cbab4cf76f0fc2907532` |
| `u_mlp_v2_mix__H128__s0` | twin | 0 | 330,955 | 0 | 3 | 0.8457 | 1.14 | 9.23 | pilot (seed 0) | `f0e629a7347a08014ae00f6d164b6158695a71118c7dabf0dfaf56e1b8a3e8f3` | `21fab0efc9c7538c08de6311d3ebdde0944f49543e17bd0487f42e50f68940fb` |
| `u_mlp_v2_mix__H128__s1` | twin | 1 | 330,955 | 1 | 4 | 0.8475 | 2.18 | 6.66 | amendment_4_2026_09_21 | `0926391ac70457ae0c7319fe1821aee5be17eb57af0879168f5b003c32effb85` | `f6cff3f3f05507908d244e37f5cef221fd7462dbc49d9f49192ed43ca0efd62d` |
| `u_mlp_v2_mix__H128__s2` | twin | 2 | 330,955 | 0 | 3 | 0.8452 | 2.04 | 5.08 | amendment_4_2026_09_21 | `d4b7157e3e272a01d815ac088b670aa99a9120c6fb3d2aa626f78fb67219a431` | `9302082f2acedc1b6ed39ed78f54d43e820584326bbd9a8320e94bfe0a9fa217` |

Compute: the replication fits took 21.71 fit-hours (one lane, 8 threads); 46.36 fit-hours spent in total against the ceiling of 150 (amendment 3 compute_ceiling_guard, unchanged). The supplement eval pass is not a fit and does not count.

## 3. The supplement eval pass

| record | dataset | models | queries | V2_GATE / V2_HELD_CONFIRMATION | lane seconds | ms / query | threads | shards | record sha256 | arrays sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| `2wiki__more_69caa3fe` | 2wiki | `u_gnn_v2_ef__H128__s1`, `u_gnn_v2_ef__H128__s2`, `u_mlp_v2_mix__H128__s1`, `u_mlp_v2_mix__H128__s2` | 12,576 | 6,290 / 6,286 | 2408 | 182.0 | 6 | 1 | `6fcd2a0e1d55c374961daeb332920664c689b59a38416c2e5829490ff1cb8e84` | `76e4e3a9abd213040c9807c635a1bce0ac183e22919ca44149bc8910df4e4afe` |
| `metaqa__more_69caa3fe` | metaqa | `u_gnn_v2_ef__H128__s1`, `u_gnn_v2_ef__H128__s2`, `u_mlp_v2_mix__H128__s1`, `u_mlp_v2_mix__H128__s2` | 39,138 | 19,738 / 19,400 | 52815 | 1338.1 | 6 | 6 | `08075559db25aa6394dbbfd52bb8c1759f115e99971e7629359bd4bf49d15334` | `07cb99f9bcb1d70bbccf7d2e55b1097e9c724b3b016f168eec88c0bb7d615a30` |
| `squad__more_69caa3fe` | squad | `u_gnn_v2_ef__H128__s1`, `u_gnn_v2_ef__H128__s2`, `u_mlp_v2_mix__H128__s1`, `u_mlp_v2_mix__H128__s2` | 11,873 | 5,841 / 6,032 | 625 | 51.7 | 6 | 1 | `926a33451561e1653a3e1b0d834a3fe12d5cd4a8f8b9e7aea29a0c9c3c7e1905` | `4f0d82a419415c5d4ffe33bfe8d8bb973f9a7454b09764eb41832885e0e81c95` |

Lane time of the pass: 15.51 lane-hours. Each supplement record is pinned to its seed-0 record (query list, half labels, fixed rrf per query); both halves are labelled as written and the pass reads nothing.

## 4. Replication on V2_GATE (seeds 0-2, the declared aggregation)

**`u_gnn_v2_ef` (GNN_GATE) — REPLICATION_PASS** (seeds present [0, 1, 2]; the mean is per query over the seeds; a family passes only if every cell holds)

| dataset | metric | slice | queries | seed 0 | seed 1 | seed 2 | mean over seeds | sd | threshold | margin | paired interval of the seed mean | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,738 | 0.9175 | 0.9051 | 0.9070 | 0.9099 | 0.0055 | 0.8840 | +0.0259 | +0.078 [+0.073, +0.082] vs `gat_universal_v1`, excludes zero: True | holds |
| metaqa | hit@1 | 3hop | 7,273 | 0.8679 | 0.8310 | 0.8536 | 0.8508 | 0.0152 | 0.8172 | +0.0336 | +0.112 [+0.103, +0.121] vs `gat_universal_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,290 | 0.8963 | 0.8996 | 0.9027 | 0.8996 | 0.0026 | 0.8758 | +0.0238 | — | holds |
| 2wiki | full_coverage@5 | all | 6,290 | 0.7674 | 0.7750 | 0.7865 | 0.7763 | 0.0079 | 0.7453 | +0.0310 | — | holds |
| squad | recall@5 | all | 5,841 | 0.8988 | 0.9048 | 0.9022 | 0.9020 | 0.0025 | 0.8996 | +0.0024 | — | holds |

Every cell holds on the seed mean.

**`u_mlp_v2_mix` (TWIN_GATE) — REPLICATION_FAIL** (seeds present [0, 1, 2]; the mean is per query over the seeds; a family passes only if every cell holds)

| dataset | metric | slice | queries | seed 0 | seed 1 | seed 2 | mean over seeds | sd | threshold | margin | paired interval of the seed mean | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,738 | 0.7772 | 0.7835 | 0.7653 | 0.7753 | 0.0075 | 0.7021 | +0.0732 | +0.160 [+0.153, +0.166] vs `qls_u_sota_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,290 | 0.8545 | 0.8523 | 0.8481 | 0.8516 | 0.0027 | 0.8578 | -0.0062 | — | FAILS |
| squad | recall@5 | all | 5,841 | 0.9105 | 0.8997 | 0.9087 | 0.9063 | 0.0047 | 0.8996 | +0.0067 | — | holds |

Cells failing on the seed mean: 2wiki recall@5 (all).

Seed mean ± sd on V2_GATE (three seeds) for every metric reported:

| dataset | arm | seeds | recall@5 | hit@1 | full_coverage@5 | mrr |
|---|---|---|---|---|---|---|
| metaqa | `u_gnn_v2_ef` | [0, 1, 2] | 0.7822 ± 0.0021 | 0.9099 ± 0.0055 | 0.6602 ± 0.0029 | 0.9369 ± 0.0035 |
| metaqa | `u_mlp_v2_mix` | [0, 1, 2] | 0.7157 ± 0.0040 | 0.7753 ± 0.0075 | 0.5880 ± 0.0051 | 0.8489 ± 0.0052 |
| 2wiki | `u_gnn_v2_ef` | [0, 1, 2] | 0.8996 ± 0.0026 | 0.8908 ± 0.0135 | 0.7763 ± 0.0078 | 0.9362 ± 0.0076 |
| 2wiki | `u_mlp_v2_mix` | [0, 1, 2] | 0.8516 ± 0.0027 | 0.8978 ± 0.0112 | 0.6721 ± 0.0034 | 0.9389 ± 0.0061 |
| squad | `u_gnn_v2_ef` | [0, 1, 2] | 0.9020 ± 0.0025 | 0.7336 ± 0.0016 | 0.9020 ± 0.0025 | 0.8089 ± 0.0011 |
| squad | `u_mlp_v2_mix` | [0, 1, 2] | 0.9063 ± 0.0047 | 0.7427 ± 0.0040 | 0.9063 ± 0.0047 | 0.8158 ± 0.0041 |

## 5. V2_HELD_CONFIRMATION (read once, REPLICATION_PASS families only)

Read 2026-09-22T18:50:06Z for ['gnn']; not read for ['twin'] (its arrays dropped unread: 3 keys). a Universal-v2 architecture-selection holdout, not a globally unseen dataset (check_1_the_halves.what_it_is_not); the confirmatory column of the replication for the families that passed on V2_GATE; nothing here is a gate.

**`u_gnn_v2_ef` (GNN_GATE) — confirmatory, not a gate** (gate-half status REPLICATION_PASS)

| dataset | metric | slice | queries | seed 0 | seed 1 | seed 2 | mean over seeds | sd | threshold | margin | paired interval of the seed mean | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | hit@1 | all | 19,400 | 0.9158 | 0.9058 | 0.9068 | 0.9095 | 0.0045 | 0.8840 | +0.0255 | +0.080 [+0.075, +0.084] vs `gat_universal_v1`, excludes zero: True | holds |
| metaqa | hit@1 | 3hop | 7,001 | 0.8620 | 0.8317 | 0.8496 | 0.8478 | 0.0124 | 0.8172 | +0.0306 | +0.112 [+0.103, +0.121] vs `gat_universal_v1`, excludes zero: True | holds |
| 2wiki | recall@5 | all | 6,286 | 0.8932 | 0.8999 | 0.9001 | 0.8977 | 0.0032 | 0.8758 | +0.0219 | — | holds |
| 2wiki | full_coverage@5 | all | 6,286 | 0.7655 | 0.7770 | 0.7840 | 0.7755 | 0.0076 | 0.7453 | +0.0302 | — | holds |
| squad | recall@5 | all | 6,032 | 0.9038 | 0.9135 | 0.9040 | 0.9071 | 0.0045 | 0.8996 | +0.0075 | — | holds |

Per seed on the held half (seed 0 copied from the pilot's held record):

| seed | metaqa hit@1 (all) | metaqa hit@1 (3hop) | 2wiki recall@5 (all) | 2wiki full_coverage@5 (all) | squad recall@5 (all) |
|---|---|---|---|---|---|
| 0 | 0.9158 | 0.8620 | 0.8932 | 0.7655 | 0.9038 |
| 1 | 0.9058 | 0.8317 | 0.8999 | 0.7770 | 0.9135 |
| 2 | 0.9068 | 0.8496 | 0.9001 | 0.7840 | 0.9040 |

## 6. Reading

ORIGINAL PILOT STATUS: PILOT_FAILED. POST-PILOT REPLICATION STATUS: GNN_REPLICATION_ONLY. the proposed universal GNN is stable enough to continue; the universal non-MP twin is not yet sufficiently robust.
The original pilot is still reported as failed; this document adds a replication reading beside it and changes no threshold, feature, architecture or population. Forbidden framings are not used: the question of the paper is how much effectiveness remains attributable to learned message passing after matching exposure and information, and this replication reads only whether two frozen arms hold their frozen cells on three seeds.

## 7. Incidents

- 2026-09-21T10:42:38Z (queued 2026-09-21T16:11:56 IST): lane R (the four authorised fits, one lane, driver v2_driver2.ps1 as the pilot's lane A) queued behind pid 10956, another session's SPLADE experiment holding 6.7 GB private on the 16 GB laptop (free 2.6 GB; a fit holds 6-8 GB private, pilot health log): the lane launches automatically when that process exits and free memory is >= 8 GB (outputs/universal_v2/logs/queue_R.log); no fit was started under memory contention; systems only, nothing scientific
- 2026-09-21T14:55:28Z (20:25:28 IST): the queue script's 1800 s memory-wait cap launched lane R although free memory was only 1.86 GB -- another session's new SPLADE process (pid 22308, 6.2 GB private) had started at 20:12 IST after the first one exited at 19:55 IST; the first fit (u_mlp_v2_mix seed 1, pid 22400) ran 4 min 40 s under that memory contention (free 0.56 GB, commit free 2.58 GB) and was stopped by hand at about 20:30 IST (2026-09-21T15:00Z) together with its driver (pid 19376) before its first epoch checkpoint existed; no fits/ artifact was written, the contended logs are kept as logs/*.contended_2026_09_21T2025.*, and the lane is relaunched from scratch once the other session's processes have exited (free 6.1 GB, commit free 11.1 GB at relaunch); the relaunched fit draws the same seeded batches, so this is systems only, nothing scientific
- 2026-09-21T15:02:48Z (20:32:48 IST): lane R relaunched by hand with the machine free (free 5.1 GB, commit free 8.8 GB, the other session's jobs gone), but another session's benchmark (pid 24172, bench_maxlen.py) had started 20 s earlier and grew to 4.7 GB private at 8-9 cores; the fit (u_mlp_v2_mix seed 1, pid 7356, set to BelowNormal priority at 20:34:41 IST) got ~1.5 cores and commit free fell to 0.85 GB, so it was stopped by hand at 20:36:02 IST (2026-09-21T15:06:02Z) with its driver (pid 27332), again before any epoch checkpoint; no fits/ artifact was written, the logs are kept as logs/*.contended_2026_09_21T2032.*; lane R is now queued by queue_fit_lane_R2.ps1, which launches only when no foreign python process holds >= 1.5 GB and free memory is >= 6 GB, with no time cap (outputs/universal_v2/logs/queue_R.log); systems only, nothing scientific
- 2026-09-22T08:30:53Z to 2026-09-22T11:52:45Z (14:00:53 to 17:22:45 IST): the laptop entered Modern Standby during epoch 3 of the last authorised fit (u_gnn_v2_ef seed 2, pid 2628) and resumed 3 h 21 m 52 s later (Kernel-Power 506/507, no reboot); the fit process survived the suspend and continued from where it was -- it consumed 1,070 CPU seconds over the whole gap (about 1 % of 8 threads) and returned to 3.6 cores after the resume, and the epoch-2 checkpoint (13:38:07 IST) was never needed. Wall-clock only: no batch was redrawn, no artifact was rewritten and the lane was not restarted; the session now holds the machine awake so the remaining fit, the eval pass and the merge are not suspended again
- 2026-09-22T15:53:50Z to 2026-09-22T16:18:55Z (21:23:50 to 21:48:55 IST): all three eval lanes of the supplement pass stopped on their SECOND metaqa shard with 'already scored on this population; a model is scored once'. The once-only guard in stage_eval scanned every <name>__more_*.json, which after the pilot's sharding convention includes the shard files of the pass being run, so shard k+1 was refused because shard k existed; the pilot never hit it because its metaqa sharding ran without --models. The guard is now shard-aware -- it ignores the files of the pass's own tag and still refuses on the seed-0 record and on any other tag's record -- with a regression test in tests/test_universal_v2_run.py (test 23). squad, 2wiki and metaqa shards 0-2 of 6 were written before the refusals and are kept; shards 3, 4 and 5 are relaunched. Systems only: no population, metric, threshold or scored number changed, and no model was scored twice

## 8. Commits since the pilot closed

- `dcbb05b Universal-v2 amendment 4: the post-pilot replication of the two frozen selected arms filed verbatim before any fit -- seeds 1-2 of u_gnn_v2_ef and u_mlp_v2_mix only, the frozen protocol and architecture, the three-seed per-query mean against the original frozen cells, REPLICATION_PASS/FAIL per family beside the untouched PILOT_FAILED; the tooling and its tests, no weight fitted`

## 9. Run record

Rendered 2026-09-22T18:50:49Z by `scripts/universal_v2_report.py --stage replication_doc` from the files below (sha256 of every sidecar a number above cites; the sidecars are gitignored, the record is committed as replication_record_<date>).

| file | sha256 |
|---|---|
| `configs/universal_v2.yaml` | `6594275eb52c42afde25003523e3940cfcb6f19908858907b0253efadcc14570` |
| `outputs/universal_v2/replication_record.json` | `9863706fa97fda934b4b8cd502c0e0bbea566db7c06befcdf42e26452c808f43` |
| `outputs/universal_v2/replication_held_record.json` | `e6c7bc3d2e229fa19982295efe5280d640e5f879c7604d7b1c197c53fba0fc75` |
| `outputs/universal_v2/held_record.json` | `862acbadaf72dbe27b0f1f691259bb36802268dc728ba5d2f04a6b43ccc27adc` |
| `outputs/universal_v2/gate_record.json` | `8dcc217b08a75723af336e8d32ced4621aeed5f4e17cb1cfdb526433497d16e5` |
| `outputs/universal_v2/selection.json` | `3ed48ae3d04334ba367cb81b8b363d32db0e2bf746e4f692d7af0936bea4e3b6` |
| `outputs/universal_v2/replication_incidents.json` | `44d6bdf01e4f6840305e8d19d562359f92b86911e303d3fbc2d8251fcfc323ff` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s0.json` | `bfd3e8e5957b259b2d8b38f85008f659f48e00eb79cdf43220aeb39817993009` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s1.json` | `6e1efe41c8c2f27bfd1d37fa26e9b8e00685e1ffb8d12f8528d489b6b3a9a60a` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s2.json` | `4cb612e195c85fde11edaec8b6f0dcd293fc706b2554cbab4cf76f0fc2907532` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s0.json` | `21fab0efc9c7538c08de6311d3ebdde0944f49543e17bd0487f42e50f68940fb` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s1.json` | `f6cff3f3f05507908d244e37f5cef221fd7462dbc49d9f49192ed43ca0efd62d` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s2.json` | `9302082f2acedc1b6ed39ed78f54d43e820584326bbd9a8320e94bfe0a9fa217` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s0.pt` | `9f749ad346afcef479f84436019771e0b940a1e647e4960908acf4105ea2ed0e` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s1.pt` | `9b3a5521babb814e83e8a0a40a94b9abd92bdee7f7f58b04f686558d0522c367` |
| `outputs/universal_v2/fits/u_gnn_v2_ef__H128__s2.pt` | `4575c41822f56d99b843aaa524a5eb407fa038ebc24a6ad04f4fef786433d4c4` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s0.pt` | `f0e629a7347a08014ae00f6d164b6158695a71118c7dabf0dfaf56e1b8a3e8f3` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s1.pt` | `0926391ac70457ae0c7319fe1821aee5be17eb57af0879168f5b003c32effb85` |
| `outputs/universal_v2/fits/u_mlp_v2_mix__H128__s2.pt` | `d4b7157e3e272a01d815ac088b670aa99a9120c6fb3d2aa626f78fb67219a431` |
| `outputs/universal_v2/eval/2wiki.json` | `ab7ad34de918fafb9fac7f46eebff014e14a741394e786b2929a3af92858890f` |
| `outputs/universal_v2/eval/2wiki.npz` | `6d8c1dcd2f815cf8c5e67664ace3ceaada3f54c6625ceeee768d35b04f13b4c2` |
| `outputs/universal_v2/eval/2wiki_query_ids.json` | `d439a33c22f39186f8e54f596aeb2bcb83a5d27745a0dc9285e8990f16923cd2` |
| `outputs/universal_v2/eval/2wiki__more_69caa3fe.json` | `6fcd2a0e1d55c374961daeb332920664c689b59a38416c2e5829490ff1cb8e84` |
| `outputs/universal_v2/eval/2wiki__more_69caa3fe.npz` | `76e4e3a9abd213040c9807c635a1bce0ac183e22919ca44149bc8910df4e4afe` |
| `outputs/universal_v2/eval/2wiki__more_69caa3fe_query_ids.json` | `d439a33c22f39186f8e54f596aeb2bcb83a5d27745a0dc9285e8990f16923cd2` |
| `outputs/universal_v2/eval/metaqa.json` | `3db26c272e8d83c92c46392647da3c7ee0747f3561afceee32bb84b2d7824676` |
| `outputs/universal_v2/eval/metaqa.npz` | `380e0288b0e048cd18d83f06defd5f794fe7b7514a741bffbfbd6289e49eab5c` |
| `outputs/universal_v2/eval/metaqa_query_ids.json` | `b491d9e90e786c1d356d2a24142ec1b810e657efdfadff85a44ff1b489618692` |
| `outputs/universal_v2/eval/metaqa__more_69caa3fe.json` | `08075559db25aa6394dbbfd52bb8c1759f115e99971e7629359bd4bf49d15334` |
| `outputs/universal_v2/eval/metaqa__more_69caa3fe.npz` | `07cb99f9bcb1d70bbccf7d2e55b1097e9c724b3b016f168eec88c0bb7d615a30` |
| `outputs/universal_v2/eval/metaqa__more_69caa3fe_query_ids.json` | `b491d9e90e786c1d356d2a24142ec1b810e657efdfadff85a44ff1b489618692` |
| `outputs/universal_v2/eval/squad.json` | `8d7778ce4168b27fab4f187354bcc2336f0080ae6771977f386899b905641111` |
| `outputs/universal_v2/eval/squad.npz` | `cda64047bea803936ce811e5fa0d006ebffa10473e7baed44f2d203bd55121f3` |
| `outputs/universal_v2/eval/squad_query_ids.json` | `3ce32ec30d8920c8e0be25c2d53e325b29e4d16a5b16f7dfc9f564eb86d8a93c` |
| `outputs/universal_v2/eval/squad__more_69caa3fe.json` | `926a33451561e1653a3e1b0d834a3fe12d5cd4a8f8b9e7aea29a0c9c3c7e1905` |
| `outputs/universal_v2/eval/squad__more_69caa3fe.npz` | `4f0d82a419415c5d4ffe33bfe8d8bb973f9a7454b09764eb41832885e0e81c95` |
| `outputs/universal_v2/eval/squad__more_69caa3fe_query_ids.json` | `3ce32ec30d8920c8e0be25c2d53e325b29e4d16a5b16f7dfc9f564eb86d8a93c` |
| `outputs/m3b/eval/metaqa.npz` | `a4739cf66a8d0eb4a751abb0e9480305986eceb3876aca7662a05c7f0dfb97dd` |
| `outputs/m3b/eval/2wiki.npz` | `d2196c2f7769512071f1a9592504be6e1bdeff9729f1b2bcfae45f1ebdd76313` |
| `outputs/m3b/eval/squad.npz` | `c14fe820295dc18374ef8f0d5775f2d422af830bc9a579f527736929aeef8385` |

