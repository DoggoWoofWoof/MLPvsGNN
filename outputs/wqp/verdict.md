# WQP: NOT_UNIFIED, KIND_FREE_TOO_DEAR, NO_GAIN

Declared in docs/WQP_POOL_PICK_UNDER_WQ.md. Carried forward: **U2s**, `own/kind_t/kt_cos:ada_f16_k0@0.85`.

- Selection set S: squad, musique, hotpotqa, 2wiki, webqsp (webqsp on its select carve, the others on s1sel); weights {'squad': 17729, 'musique': 7018, 'hotpotqa': 13335, '2wiki': 18504, 'webqsp': 2705}.
- Cap: 1545.3 (U2c's chosen pool, `g:f3+cut_cos_7000@0.9`, on S; 1953.5 under U2s's set).
- Incumbent, U2s's pick: `own/kind_t/kt_cos:ada_f16_k0@0.85` at level None on S, cost 1382.4 (within the cap).
- Pick C (kind_t, universal): none within the cap.
- Pick E (kind-free, universal): none within the cap.
- Pick O (the own construction, kind_t): none within the cap.
- U2s's verdict reproduced from the same tables: NOT_UNIFIED, KIND_FREE_TOO_DEAR, OWN_CHEAPER; `own/kind_t/kt_cos:ada_f16_k0@0.85`. The check (section 2): REPRODUCED.
- Next: U2n's zrckpt refit on U2s's pick, under WQ, in its own file; the upstream choices stand; under WQ, U2s section 10's other next forms: RRF5T's agreement as the budget, and P1h's cut over the walk's mass.

## Leave one out (each split's own training sets)

| split | left out | cap | (c) | (e) | (o) | carried |
|---|---|---|---|---|---|---|
| WQ-musique | musique | 994.1 | NOT_UNIFIED | KIND_FREE_TOO_DEAR | OWN_RAISED | O: `own/kind_t/kt_n1:ada_f6_k0@0.99` |
| WQ-hotpotqa | hotpotqa | 1714.0 | NOT_UNIFIED | KIND_FREE_TOO_DEAR | NO_GAIN | U2s: `own/kind_t/kt_cos:ada_f16_k0@0.85` |

## Each carve's ceiling and fewest rows by level

| carve | ceiling | 0.95 | 0.955 | 0.96 | 0.965 | 0.97 | 0.975 | 0.98 | 0.985 | 0.99 | 0.995 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.9947 | 1519.5 | 1666.5 | 1893.9 | 2089.3 | 2238.5 | 2978.5 | 3294.5 | 4293.8 | 5492.2 | 5492.2 | 5492.2 |
| squad | 0.9993 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 53.0 | 68.5 | 84.1 | 238.2 | 238.2 |
| musique | 0.9811 | 3470.1 | 3736.8 | 4122.4 | 4572.3 | 5343.0 | 6394.3 | 6720.0 | 6720.0 | 6720.0 | 6720.0 | 6720.0 |
| hotpotqa | 0.9881 | 314.7 | 400.4 | 467.5 | 610.0 | 985.4 | 1559.3 | 2607.6 | 3097.7 | 3097.7 | 3097.7 | 3097.7 |
| 2wiki | 0.996 | 327.2 | 366.0 | 441.3 | 480.3 | 668.0 | 841.0 | 1185.8 | 2236.4 | 3121.9 | 3624.6 | 3624.6 |
| webqsp_s1eval | 0.9607 | 5661.8 | 6981.3 | 7578.1 | 7578.1 | 7578.1 | 7578.1 | 7578.1 | 7578.1 | 7578.1 | 7578.1 | 7578.1 |
| webqsp | 0.9607 | 5680.6 | 6637.1 | 6637.1 | 6637.1 | 6637.1 | 6637.1 | 6637.1 | 6637.1 | 6637.1 | 6637.1 | 6637.1 |

## The price: the carried pick against the incumbent (ALL, rows)

| carve | carried | incumbent |
|---|---|---|
| metaqa | 0.9653, 4369.9 | 0.9653, 4369.9 |
| squad | 0.9793, 50.0 | 0.9793, 50.0 |
| musique | 0.9668, 5522.2 | 0.9668, 5522.2 |
| hotpotqa | 0.9602, 713.1 | 0.9602, 713.1 |
| 2wiki | 0.9699, 1070.5 | 0.9699, 1070.5 |
| webqsp_s1eval | 0.9069, 4803.4 | 0.9069, 4803.4 |
| webqsp | 0.9082, 4806.7 | 0.9082, 4806.7 |
