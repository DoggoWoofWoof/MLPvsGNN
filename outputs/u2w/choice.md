# U2w stage A: one pool rule on 2wiki, hotpotqa and musique

Declared in docs/U2W_ONE_POOL_RULE_THREE_DATASETS.md. Verdicts: **NOT_UNIFIED**, **KIND_FREE_TOO_DEAR**; carried: **None**.

Incumbent (pick O, per-dataset construction): `own/kind_t/kt_cos:ada_f16_k0@0.85`, level 0.96, cost 1751.9 rows on the three (P1f's weights {'musique': 7018, 'hotpotqa': 13335, '2wiki': 18504}); cap 1953.5; row price bound 2.0 x the incumbent's cost.

## ALL (rows) on s1sel

| pools | level | cost | musique | hotpotqa | 2wiki |
|---|---:|---:|---|---|---|
| pick O `own/kind_t/kt_cos:ada_f16_k0@0.85` | 0.96 | 1751.9 | 0.9668 (5522.2) | 0.9602 (713.1) | 0.9699 (1070.5) |
| carried | - | - | - | - | - |
| C's cheapest at 0.96 `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f6_k0@0.85` | 0.96 | 4215.6 | 0.9602 (4193.9) | 0.9834 (4025.2) | 0.9913 (4361.1) |
| E's cheapest at 0.96 `rrf50+FULL:h2_c25/rrf5t/kt_cos+n2_10:ada_f6_k0@0.85` | 0.96 | 4215.6 | 0.9602 (4193.9) | 0.9834 (4025.2) | 0.9913 (4361.1) |
| U2s's six-dataset C at 0.950 `rrf50+none/kind_t/kt_cos+n2_100:mass_A1all_t0.9_f4_k0@0.8` | - | 3533.5 | 0.9511 (3541.3) | 0.9834 (3221.6) | 0.99 (3755.3) |

## Frontier C (every level, past the cap too)

| level | cost | within cap | price | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---:|---|---|---|---|
| 0.95 | 3528.4 | False | 2.014 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f2_k0@0.975` | 0.9511 (3470.1) | 0.9821 (3461.8) | 0.9893 (3598.6) |
| 0.955 | 3762.2 | False | 2.147 | `dense200+none/kind_t/kt_cos+n2_10:mass_A1all_t0.99_f2_k0@0.8` | 0.955 (3736.8) | 0.9828 (3447.7) | 0.9906 (3998.5) |
| 0.96 | 4215.6 | False | 2.406 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f6_k0@0.85` | 0.9602 (4193.9) | 0.9834 (4025.2) | 0.9913 (4361.1) |
| 0.965 | 4767.3 | False | 2.721 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f4_k0@0.9` | 0.9661 (4688.6) | 0.9841 (4682.3) | 0.9933 (4858.4) |
| 0.97 | 5469.7 | False | 3.122 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:mass_A1all_t0.9_f2_k0@0.99` | 0.97 (5343.0) | 0.9854 (5348.2) | 0.994 (5605.4) |
| 0.975 | 6720.0 | False | 3.836 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:fix_7000@0.99` | 0.9759 (6720.0) | 0.9867 (6720.0) | 0.9947 (6720.0) |
| 0.98 | 6898.0 | False | 3.937 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:ada_f4_k0@0.975` | 0.9772 (6782.2) | 0.9867 (6767.2) | 0.9953 (7036.2) |
| 0.985 | 6898.0 | False | 3.937 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:ada_f4_k0@0.975` | 0.9772 (6782.2) | 0.9867 (6767.2) | 0.9953 (7036.2) |
| 0.99 | 6898.0 | False | 3.937 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:ada_f4_k0@0.975` | 0.9772 (6782.2) | 0.9867 (6767.2) | 0.9953 (7036.2) |
| 0.995 | 6898.0 | False | 3.937 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:ada_f4_k0@0.975` | 0.9772 (6782.2) | 0.9867 (6767.2) | 0.9953 (7036.2) |
| 1.0 | 6898.0 | False | 3.937 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_100:ada_f4_k0@0.975` | 0.9772 (6782.2) | 0.9867 (6767.2) | 0.9953 (7036.2) |

## Frontier E (every level, past the cap too)

| level | cost | within cap | price | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---:|---|---|---|---|
| 0.95 | 3522.1 | False | 2.01 | `rrf50+none/both/kt_cos+n2_100:mass_A1all_t0.9_f4_k0@0.8` | 0.9511 (3531.7) | 0.9814 (3218.7) | 0.9873 (3737.2) |
| 0.955 | 3762.2 | False | 2.147 | `dense200+none/rrf5t/kt_cos+n2_10:mass_A1all_t0.99_f2_k0@0.8` | 0.955 (3736.8) | 0.9828 (3447.7) | 0.9906 (3998.5) |
| 0.96 | 4215.6 | False | 2.406 | `rrf50+FULL:h2_c25/rrf5t/kt_cos+n2_10:ada_f6_k0@0.85` | 0.9602 (4193.9) | 0.9834 (4025.2) | 0.9913 (4361.1) |
| 0.965 | 4648.1 | False | 2.653 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:ada_f3_k0@0.95` | 0.9654 (4572.3) | 0.9854 (4561.6) | 0.99 (4739.1) |
| 0.97 | 5469.7 | False | 3.122 | `rrf50+FULL:h2_c25/rrf5t/kt_cos+n2_100:mass_A1all_t0.9_f2_k0@0.99` | 0.97 (5343.0) | 0.9854 (5348.2) | 0.994 (5605.4) |
| 0.975 | 6532.4 | False | 3.729 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:mass_A1all_t0.8_f8_k0@0.9` | 0.9752 (6489.4) | 0.9854 (5902.3) | 0.9947 (7002.8) |
| 0.98 | 6720.0 | False | 3.836 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:fix_7000@0.99` | 0.9772 (6720.0) | 0.9861 (6720.0) | 0.9947 (6720.0) |
| 0.985 | 6720.0 | False | 3.836 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:fix_7000@0.99` | 0.9772 (6720.0) | 0.9861 (6720.0) | 0.9947 (6720.0) |
| 0.99 | 6720.0 | False | 3.836 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:fix_7000@0.99` | 0.9772 (6720.0) | 0.9861 (6720.0) | 0.9947 (6720.0) |
| 0.995 | 6720.0 | False | 3.836 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:fix_7000@0.99` | 0.9772 (6720.0) | 0.9861 (6720.0) | 0.9947 (6720.0) |
| 1.0 | 6720.0 | False | 3.836 | `rrf50+FULL:h2_c25/both/kt_cos+n2_100:fix_7000@0.99` | 0.9772 (6720.0) | 0.9861 (6720.0) | 0.9947 (6720.0) |

## Frontier O (every level, past the cap too)

| level | cost | within cap | price | pick | musique | hotpotqa | 2wiki |
|---:|---:|---|---:|---|---|---|---|
| 0.95 | 1061.6 | True | 0.606 | `own/kind_t/kt_cos:ada_f2_k500@0.975` | 0.9511 (3537.1) | 0.9569 (516.0) | 0.9519 (516.0) |
| 0.955 | 1299.1 | True | 0.742 | `own/kind_t/kt_cos:ada_f12_k1000@0.8` | 0.9563 (4161.9) | 0.9582 (590.9) | 0.9606 (723.8) |
| 0.96 | 1478.9 | True | 0.844 | `own/kind_t/kt_cos:ada_f4_k1000@0.9` | 0.9635 (4695.2) | 0.9615 (770.0) | 0.9612 (770.0) |
| 0.965 | 1730.8 | True | 0.988 | `own/kind_t/kt_cos:ada_f3_k1000@0.975` | 0.9661 (5168.5) | 0.9675 (973.0) | 0.9686 (973.0) |
| 0.97 | 2601.5 | False | 1.485 | `own/kind_t/kt_cos:ada_f24_k0@0.9` | 0.9733 (7259.6) | 0.9702 (1202.0) | 0.9806 (1843.3) |
| 0.975 | 6111.7 | False | 3.489 | `own/kind_t/kt_cos+n2_100:mass_A1all_t0.9_f4_k0@0.9` | 0.9752 (6935.1) | 0.9847 (5349.9) | 0.9947 (6348.4) |
| 0.98 | 6395.5 | False | 3.651 | `own/kind_t/kt_cos+n2_100:mass_A1all_t0.95_f4_k0@0.9` | 0.9765 (7845.6) | 0.9847 (5421.6) | 0.9953 (6547.4) |
| 0.985 | 6395.5 | False | 3.651 | `own/kind_t/kt_cos+n2_100:mass_A1all_t0.95_f4_k0@0.9` | 0.9765 (7845.6) | 0.9847 (5421.6) | 0.9953 (6547.4) |
| 0.99 | 6395.5 | False | 3.651 | `own/kind_t/kt_cos+n2_100:mass_A1all_t0.95_f4_k0@0.9` | 0.9765 (7845.6) | 0.9847 (5421.6) | 0.9953 (6547.4) |
| 0.995 | 6395.5 | False | 3.651 | `own/kind_t/kt_cos+n2_100:mass_A1all_t0.95_f4_k0@0.9` | 0.9765 (7845.6) | 0.9847 (5421.6) | 0.9953 (6547.4) |
| 1.0 | 6395.5 | False | 3.651 | `own/kind_t/kt_cos+n2_100:mass_A1all_t0.95_f4_k0@0.9` | 0.9765 (7845.6) | 0.9847 (5421.6) | 0.9953 (6547.4) |

## Beside: the cheapest point at level 0.96 by budget form and by construction

| group | cost | pick |
|---|---:|---|
| FIX | 4252.5 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:fix_15000@0.85` |
| ADA | 4215.6 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f6_k0@0.85` |
| MASS | 4230.7 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:mass_A1all_t0.9_f4_k0@0.85` |
| rrf50+none | 4235.1 | `rrf50+none/kind_t/kt_cos+n2_10:mass_A1all_t0.95_f1_k0@1` |
| rrf50+FULL:h1_c25 | 4235.1 | `rrf50+FULL:h1_c25/kind_t/kt_cos+n2_10:mass_A1all_t0.95_f1_k0@1` |
| rrf50+FULL:h2_c25 | 4215.6 | `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f6_k0@0.85` |
| rrf50+FULL:h3_c25_v2000 | 4235.1 | `rrf50+FULL:h3_c25_v2000/kind_t/kt_cos+n2_10:mass_A1all_t0.95_f1_k0@1` |
| rrf200+none | 4378.6 | `rrf200+none/kind_t/kt_cos:mass_A1all_t0.95_f1_k0@0.99` |
| rrf200+FULL:h1_c25 | 4378.6 | `rrf200+FULL:h1_c25/kind_t/kt_cos:mass_A1all_t0.95_f1_k0@0.99` |
| rrf200+FULL:h2_c25 | 4378.6 | `rrf200+FULL:h2_c25/kind_t/kt_cos+n2_30:mass_A1all_t0.95_f1_k0@0.99` |
| rrf200+FULL:h3_c25_v2000 | 4378.6 | `rrf200+FULL:h3_c25_v2000/kind_t/kt_cos+n2_30:mass_A1all_t0.95_f1_k0@0.99` |
| dense200+none | 4438.9 | `dense200+none/both/kt_cos:fix_7000@0.9` |
| dense200+FULL:h1_c25 | 4438.9 | `dense200+FULL:h1_c25/both/kt_cos:fix_7000@0.9` |
| dense200+FULL:h2_c25 | 4373.4 | `dense200+FULL:h2_c25/both/kt_cos_r0+n2_100:mass_A1all_t0.8_f8_k0@0.8` |
| dense200+FULL:h3_c25_v2000 | 4381.2 | `dense200+FULL:h3_c25_v2000/both/kt_cos_r0+n2_100:mass_A1all_t0.8_f8_k0@0.8` |

## Beside: per dataset, the fewest mean rows any universal candidate needs (upper bound only)

| level | musique | hotpotqa | 2wiki |
|---:|---:|---:|---:|
| 0.95 | 3470.1 | 314.7 | 327.2 |
| 0.955 | 3736.8 | 400.4 | 366.0 |
| 0.96 | 4122.4 | 467.5 | 441.3 |
| 0.965 | 4572.3 | 610.0 | 480.3 |
| 0.97 | 5343.0 | 985.4 | 668.0 |
| 0.975 | 6394.3 | 1559.3 | 841.0 |
| 0.98 | 6720.0 | 2607.6 | 1185.8 |
| 0.985 | 6720.0 | 3097.7 | 2236.4 |
| 0.99 | 6720.0 | 3097.7 | 3121.9 |
| 0.995 | 6720.0 | 3097.7 | 3624.6 |
| 1.0 | 6720.0 | 3097.7 | 3624.6 |
