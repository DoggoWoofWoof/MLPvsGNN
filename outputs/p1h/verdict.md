# P1h's verdict: **NO_CUT**

at tau 0.99 s1eval's ALL falls below 0.95 on musique.

tau = 0.99 (chosen on s1sel). s1eval:

| dataset | ALL | recall | rows mean / p50 / p90 | P1g-b's rows | cut |
| --- | ---: | ---: | --- | ---: | ---: |
| metaqa | 0.9773 | 0.9944 | 3,268.1 / 3,282.0 / 3,419.0 | 3,405.1 | 4.0% |
| squad | 0.9798 | 0.9798 | 50.0 / 50.0 / 50.0 | 50.0 | 0.0% |
| musique | 0.9293 | 0.9760 | 6,194.6 / 6,359.0 / 6,882.0 | 6,494.6 | 4.6% |
| hotpotqa | 0.9911 | 0.9935 | 1,051.8 / 912.0 / 1,610.6 | 1,085.7 | 3.1% |
| 2wiki | 0.9736 | 0.9897 | 1,437.6 / 1,430.0 / 1,873.0 | 1,487.0 | 3.3% |
| webqsp (beside) | 0.9222 | 0.9705 | 4,533.1 / 4,153.0 / 6,693.0 | 4,739.5 | 4.3% |

P1f's cost: 1,886.4 against P1g-b's 1,964.2 (4.0% fewer rows).

Next: a learned cascade: a cheap scorer's per-question cut, its one setting chosen the same way on s1sel, in its own declared file.
