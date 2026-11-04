# U2y stage A: P1h's cut over the walk's mass, on musique, hotpotqa, 2wiki

Declared in docs/U2Y_WALK_MASS_CUT.md. Verdicts: **NOT_UNIFIED**, **KIND_FREE_TOO_DEAR**; carried: **None**.

Incumbent pick O `own/kind_t/kt_cos:ada_f16_k0@0.85`: level 0.96, cost 1751.9 rows; cap 1953.5; row price 2.0 x the incumbent's cost. Candidates: U2s's 1103760 universal ones, 3973536 WMS and 1103760 WMT ones.

## ALL (rows) on s1sel at the incumbent's level

| pools | cost | musique | hotpotqa | 2wiki |
|---|---:|---|---|---|
| C: `rrf50+none/kind_t/kt_cos_r0+n2_10:mass_A1all_t0.99_f1_k0@0.975~A1all.pool` | 4011.3 | 0.9602 (4156.2) | 0.9828 (3937.4) | 0.9906 (4009.5) |
| E: `rrf200+FULL:h2_c25/both/kt_cos+n2_100:mass_A1all_t0.99_f2_k0@0.95~A1all.pool` | 3897.6 | 0.9602 (4264.1) | 0.9834 (3756.3) | 0.986 (3860.5) |
| WMS: `rrf200+FULL:h2_c25/both/kt_cos+n2_100:mass_A1all_t0.99_f2_k0@0.95~A1all.pool` | 3897.6 | 0.9602 (4264.1) | 0.9834 (3756.3) | 0.986 (3860.5) |
| WMT: `rrf50+none/kind_t/kt_cos_r0+n2_100:mass_A1all_t0.95_f1_k0@eps1e-05~A1all` | 4233.4 | 0.9602 (4121.3) | 0.9834 (4154.4) | 0.9906 (4332.8) |
| WMS A1all pool: `rrf200+FULL:h2_c25/both/kt_cos+n2_100:mass_A1all_t0.99_f2_k0@0.95~A1all.pool` | 3897.6 | 0.9602 (4264.1) | 0.9834 (3756.3) | 0.986 (3860.5) |
| WMS A1all exp: `rrf50+FULL:h2_c25/both/kt_cos_r0:fix_7000@0.95~A1all.exp` | 3950.4 | 0.9602 (4227.2) | 0.9828 (3929.8) | 0.9886 (3860.2) |
| WMT A1all: `rrf50+none/kind_t/kt_cos_r0+n2_100:mass_A1all_t0.95_f1_k0@eps1e-05~A1all` | 4233.4 | 0.9602 (4121.3) | 0.9834 (4154.4) | 0.9906 (4332.8) |
| WMS KTall pool: `-` | - | - | - | - |
| WMS KTall exp: `-` | - | - | - | - |
| WMT KTall: `-` | - | - | - | - |

## frontier C (U2s's, WMS and WMT candidates)

| level | cost | within cap | form | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---|---|---|---|---|
| 0.95 | 3292.8 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_10:ada_f6_k0@0.95~A1all.pool` | 0.9505 (3525.6) | 0.9821 (3180.0) | 0.9893 (3285.9) |
| 0.955 | 3620.1 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:fix_10000@0.95~A1all.pool` | 0.955 (3743.3) | 0.9834 (3476.0) | 0.99 (3677.2) |
| 0.96 | 4011.3 | False | WMS | `rrf50+none/kind_t/kt_cos_r0+n2_10:mass_A1all_t0.99_f1_k0@0.975~A1all.pool` | 0.9602 (4156.2) | 0.9828 (3937.4) | 0.9906 (4009.5) |
| 0.965 | 4539.7 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_30:ada_f8_k0@0.95~A1all.exp` | 0.9654 (4705.6) | 0.9834 (4422.2) | 0.9933 (4561.5) |
| 0.97 | 5090.1 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.pool` | 0.97 (5221.7) | 0.9854 (4876.7) | 0.994 (5194.0) |
| 0.975 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.98 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.985 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.99 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.995 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 1.0 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |

## frontier E (U2s's, WMS and WMT candidates)

| level | cost | within cap | form | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---|---|---|---|---|
| 0.95 | 3287.7 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_30:ada_f6_k0@0.95~A1all.pool` | 0.9505 (3642.4) | 0.9801 (3179.5) | 0.9866 (3231.2) |
| 0.955 | 3518.4 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:ada_f6_k0@0.95~A1all.pool` | 0.955 (3897.7) | 0.9828 (3371.6) | 0.986 (3480.4) |
| 0.96 | 3897.6 | False | WMS | `rrf200+FULL:h2_c25/both/kt_cos+n2_100:mass_A1all_t0.99_f2_k0@0.95~A1all.pool` | 0.9602 (4264.1) | 0.9834 (3756.3) | 0.986 (3860.5) |
| 0.965 | 4500.0 | False | WMS | `rrf50+FULL:h3_c25_v2000/both/kt_cos+n2_100:ada_f4_k0@0.975~A1all.pool` | 0.9654 (4910.5) | 0.9847 (4458.1) | 0.9893 (4374.6) |
| 0.97 | 5090.1 | False | WMS | `rrf50+FULL:h2_c25/rrf5t/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.pool` | 0.97 (5221.7) | 0.9854 (4876.7) | 0.994 (5194.0) |
| 0.975 | 5957.9 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:fix_10000@0.975~A1all.exp` | 0.9759 (6368.0) | 0.9861 (5932.3) | 0.9933 (5820.9) |
| 0.98 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 0.985 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 0.99 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 0.995 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 1.0 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |

## Beside: frontier C over the walk-mass candidates alone

| level | cost | within cap | form | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---|---|---|---|---|
| 0.95 | 3292.8 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_10:ada_f6_k0@0.95~A1all.pool` | 0.9505 (3525.6) | 0.9821 (3180.0) | 0.9893 (3285.9) |
| 0.955 | 3620.1 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:fix_10000@0.95~A1all.pool` | 0.955 (3743.3) | 0.9834 (3476.0) | 0.99 (3677.2) |
| 0.96 | 4011.3 | False | WMS | `rrf50+none/kind_t/kt_cos_r0+n2_10:mass_A1all_t0.99_f1_k0@0.975~A1all.pool` | 0.9602 (4156.2) | 0.9828 (3937.4) | 0.9906 (4009.5) |
| 0.965 | 4539.7 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_30:ada_f8_k0@0.95~A1all.exp` | 0.9654 (4705.6) | 0.9834 (4422.2) | 0.9933 (4561.5) |
| 0.97 | 5090.1 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.pool` | 0.97 (5221.7) | 0.9854 (4876.7) | 0.994 (5194.0) |
| 0.975 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.98 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.985 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.99 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 0.995 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |
| 1.0 | 6641.5 | False | WMS | `rrf50+FULL:h2_c25/kind_t/kt_cos_r0+n2_100:ada_f4_k0@0.99~A1all.exp` | 0.9765 (6557.3) | 0.9867 (6483.5) | 0.9947 (6787.2) |

## Beside: frontier E over the walk-mass candidates alone

| level | cost | within cap | form | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---|---|---|---|---|
| 0.95 | 3287.7 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_30:ada_f6_k0@0.95~A1all.pool` | 0.9505 (3642.4) | 0.9801 (3179.5) | 0.9866 (3231.2) |
| 0.955 | 3518.4 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:ada_f6_k0@0.95~A1all.pool` | 0.955 (3897.7) | 0.9828 (3371.6) | 0.986 (3480.4) |
| 0.96 | 3897.6 | False | WMS | `rrf200+FULL:h2_c25/both/kt_cos+n2_100:mass_A1all_t0.99_f2_k0@0.95~A1all.pool` | 0.9602 (4264.1) | 0.9834 (3756.3) | 0.986 (3860.5) |
| 0.965 | 4500.0 | False | WMS | `rrf50+FULL:h3_c25_v2000/both/kt_cos+n2_100:ada_f4_k0@0.975~A1all.pool` | 0.9654 (4910.5) | 0.9847 (4458.1) | 0.9893 (4374.6) |
| 0.97 | 5090.1 | False | WMS | `rrf50+FULL:h2_c25/rrf5t/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.pool` | 0.97 (5221.7) | 0.9854 (4876.7) | 0.994 (5194.0) |
| 0.975 | 5957.9 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:fix_10000@0.975~A1all.exp` | 0.9759 (6368.0) | 0.9861 (5932.3) | 0.9933 (5820.9) |
| 0.98 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 0.985 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 0.99 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 0.995 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |
| 1.0 | 6140.6 | False | WMS | `rrf50+FULL:h2_c25/both/kt_cos_r0+n2_100:ada_f6_k0@0.975~A1all.exp` | 0.9772 (6545.5) | 0.9861 (5961.2) | 0.994 (6116.3) |

Ceilings: {'musique': 0.9811, 'hotpotqa': 0.9881, '2wiki': 0.996} (U2s's candidates alone: {'musique': 0.9811, 'hotpotqa': 0.9881, '2wiki': 0.996}). Kinds recorded beside: {'musique': {'passage': 1534}, 'hotpotqa': {'passage': 1508}, '2wiki': {'passage': 1496}}.

Beside, deciding nothing: the mean share of each walk's mass outside each base's B, by dataset: {'musique': {'rrf50/A1all': 0.6018, 'rrf50/KTall': 0.427, 'rrf200/A1all': 0.4931, 'rrf200/KTall': 0.2875, 'dense200/A1all': 0.5118, 'dense200/KTall': 0.3672}, 'hotpotqa': {'rrf50/A1all': 0.5823, 'rrf50/KTall': 0.3474, 'rrf200/A1all': 0.4792, 'rrf200/KTall': 0.223, 'dense200/A1all': 0.5007, 'dense200/KTall': 0.3116}, '2wiki': {'rrf50/A1all': 0.5849, 'rrf50/KTall': 0.3232, 'rrf200/A1all': 0.4907, 'rrf200/KTall': 0.2731, 'dense200/A1all': 0.5138, 'dense200/KTall': 0.3166}}.

Checks: {'rerun_equals_u2s': True, 'control_equals_u2s_table': 3311280, 'pick_o_filed_point': True, 'u2w_frontier_c_points': 11, 'weights': True}.
