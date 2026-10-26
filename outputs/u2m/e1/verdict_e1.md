# U2m E1: NO_GAIN

Declared in docs/U2M_POOLS_THREE_FAMILIES.md, section 19. Pick: `own/kind_t/kt_cos:ada_f12_k0@0.85`. The rule's pick under E1: `None` at level None, cost None. Carried: `own/kind_t/kt_cos:ada_f12_k0@0.85` at level 0.955, cost 1812.6. Cap 1953.5.

## The I and the KT order (beside)

| dataset | E1's ner slot | I ALL filed | size | I ALL E1 | size | questions with KT unchanged |
|---|---|---:|---:|---:|---:|---:|
| metaqa | IS | 0.8938 | 2021.7 | 0.8938 | 2021.7 | 1.0 |
| squad | empty | 0.9793 | 50.0 | 0.9793 | 50.0 | 0.0 |
| musique | empty | 0.7751 | 2067.2 | 0.6838 | 1199.7 | 0.0 |
| hotpotqa | empty | 0.8985 | 119.6 | 0.8985 | 119.6 | 0.0 |
| 2wiki | empty | 0.8369 | 162.2 | 0.8369 | 162.2 | 0.0 |
| webqsp | IS | 0.688 | 2119.9 | 0.682 | 2121.7 | 0.9667 |

## U2e's point on the filed families and on E1's

| dataset | filed ALL | rows | E1 ALL | rows |
|---|---:|---:|---:|---:|
| metaqa | 0.9619 | 4384.4 | 0.9619 | 4385.5 |
| squad | 0.9793 | 50.0 | 0.9793 | 50.0 |
| musique | 0.9635 | 5437.4 | 0.9153 | 4401.9 |
| hotpotqa | 0.9556 | 561.2 | 0.9529 | 561.2 |
| 2wiki | 0.9666 | 839.9 | 0.9659 | 839.9 |
| webqsp | 0.8929 | 4754.6 | 0.8922 | 4805.7 |

## The frontier on E1's families

| level | cost | within cap | pick | metaqa ALL (rows) | squad ALL (rows) | musique ALL (rows) | hotpotqa ALL (rows) | 2wiki ALL (rows) | webqsp |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.95 | | | none |  |  |  |  |  | |
| 0.955 | | | none |  |  |  |  |  | |
| 0.96 | | | none |  |  |  |  |  | |
| 0.965 | | | none |  |  |  |  |  | |
| 0.97 | | | none |  |  |  |  |  | |
| 0.975 | | | none |  |  |  |  |  | |
| 0.98 | | | none |  |  |  |  |  | |
| 0.985 | | | none |  |  |  |  |  | |
| 0.99 | | | none |  |  |  |  |  | |
| 0.995 | | | none |  |  |  |  |  | |
| 1.0 | | | none |  |  |  |  |  | |

## The frontier on the filed families

| level | cost | within cap | pick | metaqa ALL (rows) | squad ALL (rows) | musique ALL (rows) | hotpotqa ALL (rows) | 2wiki ALL (rows) | webqsp |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.95 | 1497.8 | True | `own/kind_t/kt_cos+n2_100:ada_f12_k0@0.8` | 0.9532 (3534.1) | 0.9793 (50.0) | 0.9557 (4606.8) | 0.9516 (480.2) | 0.9652 (706.6) | 0.8789 (3818.0) |
| 0.955 | 1812.6 | True | `own/kind_t/kt_cos:ada_f12_k0@0.85` | 0.9619 (4384.4) | 0.9793 (50.0) | 0.9635 (5437.4) | 0.9556 (561.2) | 0.9666 (839.9) | 0.8929 (4754.6) |
| 0.96 | 1955.1 | False | `own/kind_t/kt_cos:ada_f16_k300@0.85` | 0.9619 (4384.4) | 0.9933 (248.0) | 0.9635 (5507.1) | 0.9602 (714.3) | 0.9726 (1070.5) | 0.8929 (4811.7) |
| 0.965 | 2324.3 | False | `own/kind_t/kt_cos+n2_100:ada_f3_k1000@0.975` | 0.9706 (5310.7) | 0.9967 (973.0) | 0.9654 (5168.5) | 0.9655 (973.0) | 0.9713 (973.0) | 0.8982 (5305.2) |
| 0.97 | 3364.3 | False | `own/kind_t/kt_cos:ada_f24_k0@0.95` | 0.9893 (7308.8) | 0.9793 (50.0) | 0.9752 (9485.9) | 0.9708 (1439.1) | 0.9833 (2249.2) | 0.9188 (8293.6) |
| 0.975 | 4542.4 | False | `own/kind_t/kt_cos:ada_f24_k1000@1` | 0.996 (9428.0) | 0.9967 (1050.0) | 0.9759 (12427.5) | 0.9755 (1763.3) | 0.9866 (2743.4) | 0.9568 (10807.3) |
| 0.98 | 6800.2 | False | `own/kind_t/kt_cos+n2_30:fix_15000@0.95` | 0.9893 (7399.2) | 0.9993 (5698.4) | 0.9759 (9286.0) | 0.9847 (6168.7) | 0.9953 (6858.5) | 0.9175 (7853.7) |
| 0.985 | 6800.2 | False | `own/kind_t/kt_cos+n2_30:fix_15000@0.95` | 0.9893 (7399.2) | 0.9993 (5698.4) | 0.9759 (9286.0) | 0.9847 (6168.7) | 0.9953 (6858.5) | 0.9175 (7853.7) |
| 0.99 | 7404.2 | False | `own/kind_t/kt_cos:fix_15000@0.975` | 0.9927 (8301.7) | 0.9993 (6340.2) | 0.9759 (10250.4) | 0.9847 (6658.0) | 0.9947 (7118.1) | 0.9348 (8818.9) |
| 0.995 | 7404.2 | False | `own/kind_t/kt_cos:fix_15000@0.975` | 0.9927 (8301.7) | 0.9993 (6340.2) | 0.9759 (10250.4) | 0.9847 (6658.0) | 0.9947 (7118.1) | 0.9348 (8818.9) |
| 1.0 | 7404.2 | False | `own/kind_t/kt_cos:fix_15000@0.975` | 0.9927 (8301.7) | 0.9993 (6340.2) | 0.9759 (10250.4) | 0.9847 (6658.0) | 0.9947 (7118.1) | 0.9348 (8818.9) |

## Misses by cause (beside)

| dataset | candidate | families | missed | TAU_CUT | BUDGET | UNREACHED |
|---|---|---|---:|---:|---:|---:|
| metaqa | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_filed | 329 | 287 | 8 | 34 |
| metaqa | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_E1 | 329 | 287 | 8 | 34 |
| squad | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_filed | 31 | 0 | 30 | 1 |
| squad | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_E1 | 31 | 0 | 30 | 1 |
| musique | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_filed | 62 | 22 | 9 | 31 |
| musique | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_E1 | 141 | 44 | 67 | 30 |
| hotpotqa | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_filed | 69 | 13 | 38 | 18 |
| hotpotqa | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_E1 | 73 | 14 | 44 | 15 |
| 2wiki | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_filed | 50 | 13 | 31 | 6 |
| 2wiki | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_E1 | 51 | 12 | 33 | 6 |
| webqsp | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_filed | 2077 | 664 | 369 | 1044 |
| webqsp | `own/kind_t/kt_cos:ada_f12_k0@0.85` | u2e_pick_E1 | 2090 | 672 | 382 | 1036 |
