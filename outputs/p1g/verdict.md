# P1g's verdict: **TARGET_MET**

Chosen on s1sel: `f3+beam1_res_P200_m10`; today's pool is `f1+none`. Targets: min(0.95, ceiling - 0.005).

| dataset | carve | target | today ALL / rows | chosen ALL / rows |
| --- | --- | ---: | --- | --- |
| metaqa | s1sel | 0.9500 | 0.9519 / 2,022 | 0.9886 / 6,184 |
| squad | s1sel | 0.9500 | 0.9793 / 50 | 0.9880 / 298 |
| musique | s1sel | 0.9500 | 0.8846 / 2,094 | 0.9537 / 6,280 |
| hotpotqa | s1sel | 0.9500 | 0.9556 / 94 | 0.9861 / 478 |
| 2wiki | s1sel | 0.9500 | 0.9525 / 105 | 0.9693 / 573 |
| webqsp | s1eval (beside) | | 0.8410 / 2,120 | 0.9215 / 6,314 |

Cost (the question-weighted mean of the five datasets' mean rows): chosen 2,263.4, today 699.7.

Beside: the cheapest candidate meeting every target, per family:

| family | candidate | cost |
| --- | --- | ---: |
| f3:beam1_res | `f3+beam1_res_P200_m10` | 2,263.4 |
| f4:beam1_res | `f4+beam1_res_P200_m1` | 2,345.8 |
| f4:cut_res | `f4+cut_res_50` | 2,351.4 |
| f4:beam1_cos | `f4+beam1_cos_P10_m10` | 2,359.0 |
| f3:cut_res | `f3+cut_res_500` | 2,365.4 |
| f4:beam2_cos | `f4+beam2_cos_P20_m2` | 2,369.8 |
| f1.5:cut_cos | `f1.5+cut_cos_5000` | 2,421.7 |
| f4:cut_rrf | `f4+cut_rrf_200` | 2,446.8 |
| f4:cut_w | `f4+cut_w_200` | 2,450.7 |
| f2:cut_rrf | `f2+cut_rrf_3000` | 2,521.2 |
| f1.5:cut_rrf | `f1.5+cut_rrf_7000` | 2,524.6 |
| f1.5:cut_res | `f1.5+cut_res_7000` | 2,530.8 |
| f2:cut_cos | `f2+cut_cos_3000` | 2,539.6 |
| f2:cut_res | `f2+cut_res_3000` | 2,546.1 |
| f4:beam2_res | `f4+beam2_res_P20_m5` | 2,569.9 |
| f3:cut_cos | `f3+cut_cos_1000` | 2,670.4 |
| f4:cut_cos | `f4+cut_cos_500` | 2,689.7 |
| f2:cut_w | `f2+cut_w_5000` | 2,700.3 |
| f3:cut_rrf | `f3+cut_rrf_2000` | 2,989.3 |
| f3:cut_w | `f3+cut_w_2000` | 3,005.5 |

Beside: each dataset's cheapest candidate reaching its own target (a per-dataset pick, an upper bound only):

| dataset | candidate | ALL | rows |
| --- | --- | ---: | ---: |
| metaqa | `f1+none` | 0.9519 | 2,022 |
| squad | `f0.25+none` | 0.9793 | 50 |
| musique | `f2+cut_rrf_3000` | 0.9505 | 5,612 |
| hotpotqa | `f0.5+beam2_res_P10_m1` | 0.9503 | 86 |
| 2wiki | `f1+none` | 0.9525 | 105 |
