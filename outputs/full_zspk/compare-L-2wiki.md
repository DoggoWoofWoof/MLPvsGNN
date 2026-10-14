# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\full_zspg\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7799 | 0.8045 | +0.0247 [+0.0217, +0.0276] | GAIN | 0.0047 | +0.0210 | +0.0351 |
| squad | in-domain | 11873 | 0.9103 | 0.9107 | +0.0004 [-0.0016, +0.0024] | WITHIN | 0.9054 | +0.0004 | -0.0026 |
| musique | in-domain | 2417 | 0.5684 | 0.5755 | +0.0070 [-0.0003, +0.0147] | WITHIN | 0.4734 | -0.0012 | +0.0281 |
| hotpotqa | in-domain | 7405 | 0.9016 | 0.9049 | +0.0032 [+0.0001, +0.0063] | WITHIN | 0.6846 | +0.0043 | -0.0004 |
| 2wiki | zero-shot | 12576 | 0.7937 | 0.8601 | +0.0663 [+0.0631, +0.0695] | GAIN | 0.6079 | +0.0842 | +0.0757 |
| webqsp | zero-shot | 1503 | 0.2633 | 0.2983 | +0.0351 [+0.0170, +0.0529] | GAIN | 0.0535 | +0.0273 | +0.0346 |

## Against outputs\step1\fits\L-2wiki (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.8045 | +0.1577 [+0.1519, +0.1633] | GAIN | 0.0047 | +0.1595 | +0.2981 |
| squad | in-domain | 11873 | 0.9112 | 0.9107 | -0.0005 [-0.0027, +0.0016] | WITHIN | 0.9054 | -0.0005 | -0.0047 |
| musique | in-domain | 2417 | 0.5601 | 0.5755 | +0.0153 [+0.0074, +0.0227] | GAIN | 0.4734 | +0.0112 | +0.0294 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9049 | +0.0072 [+0.0036, +0.0105] | WITHIN | 0.6846 | +0.0111 | +0.0096 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.8601 | +0.0527 [+0.0496, +0.0558] | GAIN | 0.6079 | +0.0604 | +0.0869 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2983 | +0.1908 [+0.1666, +0.2157] | GAIN | 0.0535 | +0.1497 | +0.1171 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
