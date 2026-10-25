# U2m (a): RELATIONAL_CHEAPER

Declared in docs/U2M_POOLS_THREE_FAMILIES.md, section 14. Carried forward: `own/kind_t/kt_cos:ada_f12_k0@0.85|G0|RWe@0.5` at level 0.955, cost 1809.6 (cap 1953.5).

(b) carried `own/kind_t/kt_cos:ada_f12_k0@0.85|G0` at level 0.955, cost 1812.6. 56 candidates: 8 points of the carried frontier at G = 0 x ['beta0', 'RWr@0.5', 'RWr@1', 'RWr@2', 'RWe@0.5', 'RWe@1', 'RWe@2'].

## The frontier

| level | pick | cost | within the cap |
|---:|---|---:|---|
| 0.95 | `own/kind_t/kt_cos+n2_100:ada_f12_k0@0.8|G0|RWe@0.5` | 1495.1 | True |
| 0.955 | `own/kind_t/kt_cos:ada_f12_k0@0.85|G0|RWe@0.5` | 1809.6 | True |
| 0.96 | `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|beta0` | 1955.1 | False |
| 0.965 | `own/kind_t/kt_cos+n2_100:ada_f3_k1000@0.975|G0|beta0` | 2324.3 | False |
| 0.97 | `own/kind_t/kt_cos:ada_f24_k0@0.95|G0|RWe@0.5` | 3358.7 | False |
| 0.975 | `own/kind_t/kt_cos:ada_f24_k1000@1|G0|beta0` | 4542.4 | False |
| 0.98 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWe@0.5` | 6799.7 | False |
| 0.985 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWe@0.5` | 6799.7 | False |
| 0.99 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWr@0.5` | 6806.1 | False |
| 0.995 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWr@0.5` | 6806.1 | False |
| 1.0 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWr@0.5` | 6806.1 | False |

## At the carried pick's point (ALL / rows; webqsp is s1eval)

| configuration | cost | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---:|---|---|---|---|---|---|
| beta0 | 1812.6 | 0.9619 / 4384.4 | 0.9793 / 50.0 | 0.9635 / 5437.4 | 0.9556 / 561.2 | 0.9666 / 839.9 | 0.8929 / 4754.6 |
| RWr@0.5 | 1812.9 | 0.988 / 4384.9 | 0.9793 / 50.0 | 0.9654 / 5439.4 | 0.9569 / 561.2 | 0.9652 / 839.9 | 0.9208 / 4759.3 |
| RWr@1 | 1817.4 | 0.99 / 4396.8 | 0.9793 / 50.0 | 0.9674 / 5459.6 | 0.9556 / 561.2 | 0.9619 / 839.9 | 0.9308 / 4794.4 |
| RWr@2 | 1793.5 | 0.9733 / 4255.5 | 0.9793 / 50.0 | 0.9654 / 5530.0 | 0.9463 / 561.2 | 0.9525 / 839.9 | 0.9215 / 4848.2 |
| RWe@0.5 | 1809.6 | 0.9653 / 4369.9 | 0.9793 / 50.0 | 0.9654 / 5439.4 | 0.9569 / 561.2 | 0.9652 / 839.9 | 0.9069 / 4747.0 |
| RWe@1 | 1810.6 | 0.9439 / 4365.4 | 0.9793 / 50.0 | 0.9674 / 5459.6 | 0.9556 / 561.2 | 0.9619 / 839.9 | 0.9082 / 4768.3 |
| RWe@2 | 1800.9 | 0.8584 / 4289.7 | 0.9793 / 50.0 | 0.9654 / 5530.0 | 0.9463 / 561.2 | 0.9525 / 839.9 | 0.8809 / 4841.4 |

## The pick's misses against the carried pick's

| dataset | carried missed | pick missed | recovered (TAU_CUT/BUDGET/UNREACHED) | lost |
|---|---:|---:|---|---|
| metaqa | 329 | 207 | 141/0/1 | 20/0/0 |
| squad | 31 | 31 | 0/0/0 | 0/0/0 |
| musique | 62 | 59 | 1/2/0 | 0/0/0 |
| hotpotqa | 69 | 67 | 3/0/0 | 1/0/0 |
| 2wiki | 50 | 52 | 2/0/0 | 4/0/0 |
| webqsp | 2077 | 2011 | 132/14/4 | 84/0/0 |

## The types (beside)

| dataset | EM steps | converged | mean cos | component sizes | null entries | NMI with relation |
|---|---:|---|---:|---|---:|---:|
| metaqa | 25 | False | 0.517403 | [605, 12450] | 11 | 0.355272 |
| squad | 25 | False | 0.285723 | [3385, 11547] | 0 | None |
| musique | 25 | False | 0.340156 | [2174, 12130] | 0 | None |
| hotpotqa | 25 | False | 0.387211 | [1877, 13415] | 1454 | None |
| 2wiki | 25 | False | 0.421548 | [1452, 13150] | 1186 | None |
| webqsp | 25 | False | 0.511359 | [1753, 13343] | 51584 | 0.479945 |

webqsp's s1eval is beside; it never moves the choice.
