# Screen scr-u2d-new-zrckpv (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.7986 | -0.0022 [-0.0037, -0.0007] | WITHIN | 0.0047 | -0.0014 | -0.0100 |
| squad | in-domain | 11873 | 0.9121 | 0.9114 | -0.0007 [-0.0029, +0.0017] | WITHIN | 0.9054 | -0.0007 | +0.0037 |
| musique | zero-shot | 2417 | 0.5153 | 0.5195 | +0.0043 [-0.0031, +0.0113] | WITHIN | 0.4734 | +0.0145 | +0.0079 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.8450 | -0.0605 [-0.0652, -0.0560] | LOSS | 0.6846 | -0.1194 | -0.0066 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9147 | -0.0198 [-0.0227, -0.0168] | LOSS | 0.6079 | -0.0507 | +0.0042 |
| webqsp | zero-shot | 1503 | 0.3561 | 0.3729 | +0.0168 [+0.0035, +0.0296] | GAIN | 0.0535 | +0.0193 | +0.0120 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
