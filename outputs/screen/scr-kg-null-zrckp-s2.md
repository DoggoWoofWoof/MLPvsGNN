# Screen scr-kg-base-zrckp-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7995 | 0.8038 | +0.0043 [+0.0030, +0.0055] | WITHIN | 0.0047 | +0.0046 | +0.0150 |
| squad | in-domain | 11873 | 0.9116 | 0.9106 | -0.0009 [-0.0028, +0.0009] | WITHIN | 0.9054 | -0.0009 | +0.0033 |
| musique | zero-shot | 2417 | 0.5129 | 0.5117 | -0.0012 [-0.0060, +0.0032] | WITHIN | 0.4734 | -0.0033 | -0.0058 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9055 | +0.0014 [-0.0009, +0.0038] | WITHIN | 0.6846 | +0.0042 | -0.0092 |
| 2wiki | in-domain | 12576 | 0.9319 | 0.9381 | +0.0062 [+0.0045, +0.0078] | WITHIN | 0.6079 | +0.0164 | -0.0007 |
| webqsp | zero-shot | 1503 | 0.3570 | 0.3448 | -0.0121 [-0.0227, -0.0025] | LOSS | 0.0535 | -0.0113 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
