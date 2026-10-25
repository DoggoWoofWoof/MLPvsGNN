# U2p: LINKWEIGHT_RAISED

Declared in docs/U2P_LINK_SCORE_WALK.md. Carried forward: `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5` at level 0.96, cost 1953.4 (cap 1953.5).

The carried pick was `own/kind_t/kt_cos:ada_f12_k0@0.85|G0|RWe@0.5` at level 0.955, cost 1809.6 ((a) RELATIONAL_CHEAPER: RWe at beta 0.5). 32 candidates: 8 points of the carried frontier at G = 0 on RWe@0.5 x ['beta0', 'LW@0.5', 'LW@1', 'LW@2']. The KBs (metaqa, webqsp) have no score: their candidates are (a)'s filed ones at its pick at every beta.

## The frontier

| level | pick | cost | within the cap |
|---:|---|---:|---|
| 0.95 | `own/kind_t/kt_cos+n2_100:ada_f12_k0@0.8|G0|RWe@0.5|beta0` | 1495.1 | True |
| 0.955 | `own/kind_t/kt_cos:ada_f12_k0@0.85|G0|RWe@0.5|beta0` | 1809.6 | True |
| 0.96 | `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5` | 1953.4 | True |
| 0.965 | `own/kind_t/kt_cos+n2_100:ada_f3_k1000@0.975|G0|RWe@0.5|LW@0.5` | 2324.3 | False |
| 0.97 | `own/kind_t/kt_cos:ada_f24_k0@0.95|G0|RWe@0.5|beta0` | 3358.7 | False |
| 0.975 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWe@0.5|beta0` | 6799.7 | False |
| 0.98 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWe@0.5|beta0` | 6799.7 | False |
| 0.985 | `own/kind_t/kt_cos+n2_30:fix_15000@0.95|G0|RWe@0.5|beta0` | 6799.7 | False |
| 0.99 | `own/kind_t/kt_cos:fix_15000@0.975|G0|RWe@0.5|beta0` | 7403.4 | False |
| 0.995 | `own/kind_t/kt_cos:fix_15000@0.975|G0|RWe@0.5|beta0` | 7403.4 | False |
| 1.0 | `own/kind_t/kt_cos:fix_15000@0.975|G0|RWe@0.5|beta0` | 7403.4 | False |

## At the carried pick's point (ALL / rows; webqsp is s1eval)

| configuration | cost | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---:|---|---|---|---|---|---|
| beta0 | 1809.6 | 0.9653 / 4369.9 | 0.9793 / 50.0 | 0.9654 / 5439.4 | 0.9569 / 561.2 | 0.9652 / 839.9 | 0.9069 / 4747.0 |
| LW@0.5 | 1810.9 | 0.9653 / 4369.9 | 0.9793 / 50.0 | 0.9668 / 5452.8 | 0.9576 / 561.2 | 0.9639 / 839.9 | 0.9069 / 4747.0 |
| LW@1 | 1813.1 | 0.9653 / 4369.9 | 0.9793 / 50.0 | 0.9681 / 5475.4 | 0.9556 / 561.2 | 0.9626 / 839.9 | 0.9069 / 4747.0 |
| LW@2 | 1818.7 | 0.9653 / 4369.9 | 0.9793 / 50.0 | 0.9648 / 5532.9 | 0.9516 / 561.2 | 0.9559 / 839.9 | 0.9069 / 4747.0 |

## The pick's misses against the carried pick's

| dataset | carried missed | pick missed | recovered (TAU_CUT/BUDGET/UNREACHED) | lost |
|---|---:|---:|---|---|
| metaqa | 207 | 207 | 0/0/0 | 0/0/0 |
| squad | 31 | 10 | 0/21/0 | 0/0/0 |
| musique | 59 | 57 | 1/1/1 | 1/0/0 |
| hotpotqa | 67 | 62 | 3/2/0 | 0/0/0 |
| 2wiki | 52 | 45 | 7/0/0 | 0/0/0 |
| webqsp | 2011 | 1959 | 52/0/0 | 0/0/0 |

## KIND_T's order moved (share of questions, beside)

| dataset | LW@0.5 | LW@1 | LW@2 |
|---|---:|---:|---:|
| musique | 1.0 | 1.0 | 1.0 |
| squad | 1.0 | 1.0 | 1.0 |
| hotpotqa | 1.0 | 1.0 | 1.0 |
| 2wiki | 1.0 | 1.0 | 1.0 |

## The score types (beside)

| dataset | scorer | entries | null share | zbar lowest | zbar highest | type sizes | K1 equal |
|---|---|---:|---:|---:|---:|---|---|
| musique | A | 4687840 | 0.0524 | -2.2627 | 3.5457 | [69278, 69534] | True |
| squad | A | 1756400 | 0.3573 | -2.1684 | 2.9909 | [17488, 17778] | False |
| hotpotqa | A | 64318458 | 0.0 | -1.9338 | 3.6546 | [1004890, 1005062] | True |
| 2wiki | H | 142967492 | 0.0 | -2.0988 | 3.4707 | [2233812, 2233954] | True |

webqsp's s1eval is beside; it never moves the choice.
