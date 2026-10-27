# Screen scr-p1gc-zspxr (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8004 | 0.8027 | +0.0023 [+0.0011, +0.0035] | WITHIN | 0.0047 | +0.0030 | +0.0124 |
| squad | in-domain | 11873 | 0.9129 | 0.9119 | -0.0010 [-0.0028, +0.0008] | WITHIN | 0.9054 | -0.0010 | -0.0054 |
| musique | zero-shot | 2417 | 0.5169 | 0.5127 | -0.0041 [-0.0102, +0.0019] | WITHIN | 0.4734 | -0.0004 | -0.0368 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9074 | -0.0013 [-0.0036, +0.0010] | WITHIN | 0.6846 | -0.0001 | +0.0280 |
| 2wiki | in-domain | 12576 | 0.9344 | 0.9307 | -0.0038 [-0.0053, -0.0022] | WITHIN | 0.6079 | -0.0078 | -0.0072 |
| webqsp | zero-shot | 1503 | 0.3354 | 0.3338 | -0.0016 [-0.0122, +0.0092] | WITHIN | 0.0535 | -0.0007 | -0.0126 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8027 | +0.1489 [+0.1433, +0.1541] | GAIN | 0.0047 | +0.1486 | +0.2913 |
| squad | in-domain | 11873 | 0.9100 | 0.9119 | +0.0019 [-0.0003, +0.0041] | WITHIN | 0.9054 | +0.0019 | -0.0050 |
| musique | zero-shot | 2417 | 0.2696 | 0.5127 | +0.2431 [+0.2305, +0.2552] | GAIN | 0.4734 | +0.1320 | +0.3806 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9074 | +0.0059 [+0.0025, +0.0091] | WITHIN | 0.6846 | +0.0095 | +0.0122 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9307 | +0.0575 [+0.0544, +0.0607] | GAIN | 0.6079 | +0.1159 | +0.0639 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3338 | +0.1559 [+0.1321, +0.1804] | GAIN | 0.0535 | +0.1204 | +0.1111 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
