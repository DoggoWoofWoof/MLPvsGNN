# U2m (b): NO_GAIN

Declared in docs/U2M_POOLS_THREE_FAMILIES.md, section 11. Carried forward: `own/kind_t/kt_cos:ada_f12_k0@0.85|G0` at level 0.955, cost 1812.6 (cap 1953.5).

(c) and (e) carried U2e: `own/kind_t/kt_cos:ada_f12_k0@0.85` at level 0.955, cost 1812.6. 56 candidates: 8 points of its frontier x G in [0, 10, 25, 50, 100, 200, 500].

## The frontier

| level | pick | cost | within the cap |
|---:|---|---:|---|
| 0.95 | `own/kind_t/kt_cos+n2_100:ada_f12_k0@0.8|G0` | 1497.8 | True |
| 0.955 | `own/kind_t/kt_cos:ada_f12_k0@0.85|G0` | 1812.6 | True |
| 0.96 | `own/kind_t/kt_cos:ada_f16_k300@0.85|G0` | 1955.1 | False |
| 0.965 | `own/kind_t/kt_cos+n2_100:ada_f3_k1000@0.975|G0` | 2324.3 | False |
| 0.97 | `own/kind_t/kt_cos:ada_f24_k0@0.95|G0` | 3364.3 | False |
| 0.975 | `own/kind_t/kt_cos:ada_f24_k1000@1|G0` | 4542.4 | False |
| 0.98 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0` | 6800.2 | False |
| 0.985 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0` | 6800.2 | False |
| 0.99 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G10` | 7288.5 | False |
| 0.995 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G10` | 7288.5 | False |
| 1.0 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G10` | 7288.5 | False |

## The carried pick's misses that each G recovers

| dataset | missed | G=10 | G=25 | G=50 | G=100 | G=200 | G=500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 329 | 174 | 175 | 175 | 175 | 175 | 175 |
| squad | 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| musique | 62 | 2 | 5 | 8 | 12 | 13 | 16 |
| hotpotqa | 69 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2wiki | 50 | 1 | 2 | 2 | 2 | 2 | 2 |
| webqsp | 2077 | 257 | 305 | 334 | 411 | 412 | 437 |

webqsp's s1eval is beside; it never moves the choice.
