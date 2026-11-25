# SEM2 R3(c) report

Declared in docs/SEM2_EDGE_OPERATORS.md section 11. Verdict: **PARTIAL**.

| dataset | test relations | Delta = name - cosine | Gain from the name | call | d_r > 0 | mean MRR cosine / mean name / name / shuffled |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 7 | -0.12 [-0.54, +0.24] LEVEL | +0.24 [+0.04, +0.45] ABOVE | NO | 2 | 0.58 / 0.23 / 0.46 / 0.40 |
| webqsp | 163 | +11.98 [+8.99, +15.23] ABOVE | +6.09 [+3.78, +8.77] ABOVE | TRANSFERS | 126 | 2.63 / 8.51 / 14.60 / 9.66 |

Expectations: X18 WebQSP Delta ABOVE: True; X19 WebQSP gain from the name ABOVE: True; X21 WebQSP shuffled name below the name net: True; X20 MetaQA Delta not ABOVE: True
