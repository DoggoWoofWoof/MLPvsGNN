# Screen scr-a6ex-zrckprn-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8032 | -0.0009 [-0.0016, -0.0001] | WITHIN | 0.0047 | -0.0008 | -0.0014 |
| squad | in-domain | 11873 | 0.9114 | 0.9129 | +0.0015 [-0.0005, +0.0036] | WITHIN | 0.9054 | +0.0015 | +0.0028 |
| musique | in-domain | 2417 | 0.5779 | 0.5801 | +0.0022 [-0.0027, +0.0074] | WITHIN | 0.4734 | +0.0087 | -0.0021 |
| hotpotqa | zero-shot | 7405 | 0.8558 | 0.8619 | +0.0061 [+0.0032, +0.0089] | WITHIN | 0.6846 | +0.0107 | +0.0041 |
| 2wiki | in-domain | 12576 | 0.9331 | 0.9358 | +0.0026 [+0.0011, +0.0041] | WITHIN | 0.6079 | +0.0088 | -0.0073 |
| webqsp | zero-shot | 1503 | 0.3411 | 0.3526 | +0.0115 [+0.0025, +0.0215] | GAIN | 0.0535 | +0.0100 | +0.0040 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8032 | +0.1492 [+0.1436, +0.1546] | GAIN | 0.0047 | +0.1526 | +0.2839 |
| squad | in-domain | 11873 | 0.9116 | 0.9129 | +0.0013 [-0.0011, +0.0036] | WITHIN | 0.9054 | +0.0013 | +0.0054 |
| musique | in-domain | 2417 | 0.5550 | 0.5801 | +0.0251 [+0.0173, +0.0329] | GAIN | 0.4734 | +0.0306 | +0.0319 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8619 | +0.0034 [-0.0013, +0.0077] | WITHIN | 0.6846 | +0.0051 | -0.0211 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9358 | +0.0685 [+0.0652, +0.0720] | GAIN | 0.6079 | +0.1438 | +0.0551 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3526 | +0.2326 [+0.2081, +0.2572] | GAIN | 0.0535 | +0.1896 | +0.0685 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
