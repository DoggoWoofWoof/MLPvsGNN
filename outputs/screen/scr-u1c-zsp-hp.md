# Screen scr-u1c-zsp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7807 | 0.7863 | +0.0056 [+0.0036, +0.0076] | WITHIN | 0.0047 | +0.0052 | -0.0044 |
| squad | in-domain | 11873 | 0.9109 | 0.9121 | +0.0012 [-0.0010, +0.0034] | WITHIN | 0.9054 | +0.0012 | -0.0033 |
| musique | in-domain | 2417 | 0.5647 | 0.5613 | -0.0034 [-0.0103, +0.0032] | WITHIN | 0.4734 | -0.0012 | -0.0017 |
| hotpotqa | zero-shot | 7405 | 0.8435 | 0.8038 | -0.0397 [-0.0446, -0.0350] | LOSS | 0.6846 | -0.0785 | -0.0081 |
| 2wiki | in-domain | 12576 | 0.8744 | 0.8605 | -0.0139 [-0.0172, -0.0105] | LOSS | 0.6079 | -0.0353 | -0.0056 |
| webqsp | zero-shot | 1503 | 0.2262 | 0.2364 | +0.0102 [-0.0041, +0.0241] | WITHIN | 0.0535 | +0.0067 | -0.0180 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7863 | +0.1323 [+0.1268, +0.1377] | GAIN | 0.0047 | +0.1393 | +0.2518 |
| squad | in-domain | 11873 | 0.9116 | 0.9121 | +0.0004 [-0.0021, +0.0029] | WITHIN | 0.9054 | +0.0004 | -0.0014 |
| musique | in-domain | 2417 | 0.5550 | 0.5613 | +0.0063 [-0.0002, +0.0126] | WITHIN | 0.4734 | +0.0174 | +0.0046 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8038 | -0.0548 [-0.0598, -0.0498] | LOSS | 0.6846 | -0.1070 | -0.0043 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8605 | -0.0067 [-0.0101, -0.0033] | WITHIN | 0.6079 | -0.0134 | -0.0024 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2364 | +0.1164 [+0.0930, +0.1384] | GAIN | 0.0535 | +0.0878 | +0.0459 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
