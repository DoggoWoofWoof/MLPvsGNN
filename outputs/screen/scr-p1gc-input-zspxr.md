# Screen scr-p1gc-zspxr (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gc-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7992 | 0.8027 | +0.0035 [+0.0025, +0.0045] | WITHIN | 0.0047 | +0.0038 | +0.0115 |
| squad | in-domain | 11873 | 0.9131 | 0.9119 | -0.0012 [-0.0028, +0.0005] | WITHIN | 0.9054 | -0.0012 | -0.0048 |
| musique | zero-shot | 2417 | 0.5206 | 0.5127 | -0.0079 [-0.0134, -0.0021] | LOSS | 0.4734 | -0.0012 | +0.0128 |
| hotpotqa | in-domain | 7405 | 0.9105 | 0.9074 | -0.0031 [-0.0055, -0.0009] | WITHIN | 0.6846 | -0.0023 | +0.0012 |
| 2wiki | in-domain | 12576 | 0.9289 | 0.9307 | +0.0018 [+0.0002, +0.0034] | WITHIN | 0.6079 | +0.0033 | -0.0037 |
| webqsp | zero-shot | 1503 | 0.3513 | 0.3338 | -0.0175 [-0.0272, -0.0085] | LOSS | 0.0535 | -0.0166 | -0.0106 |

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
