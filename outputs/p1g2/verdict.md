# P1g2's verdict: **TARGET_MET**

Chosen on s1sel: `g:f1.5+cut_cos_5000`; today's pool is `f1+none`. Targets: min(0.95, ceiling - 0.005).

| dataset | carve | target | today ALL / rows | chosen ALL / rows |
| --- | --- | ---: | --- | --- |
| metaqa | s1sel | 0.9500 | 0.9519 / 2,022 | 0.9646 / 3,406 |
| squad | s1sel | 0.9500 | 0.9793 / 50 | 0.9793 / 50 |
| musique | s1sel | 0.9500 | 0.8846 / 2,094 | 0.9505 / 6,600 |
| hotpotqa | s1sel | 0.9500 | 0.9556 / 94 | 0.9914 / 1,091 |
| 2wiki | s1sel | 0.9500 | 0.9525 / 105 | 0.9693 / 1,486 |
| webqsp | s1eval (beside) | | 0.8410 / 2,120 | 0.9328 / 4,740 |

Cost (P1f's question-weighted mean rows): chosen 1,975.5, today 699.7.
P1g's own choice: `f3+beam1_res_P200_m10` (TARGET_MET, cost 2,263.4).

Questions KIND_T does not expand (today's fill 0), where the gate drops the extra set:

| dataset | questions | fill 0 |
| --- | ---: | ---: |
| metaqa | 1,497 | 0 |
| squad | 1,498 | 1,498 |
| musique | 1,534 | 0 |
| hotpotqa | 1,508 | 0 |
| 2wiki | 1,496 | 0 |
| webqsp (s1eval, beside) | 1,503 | 0 |
