# Screen scr-kg-base-zrckp-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7995 | 0.8036 | +0.0041 [+0.0031, +0.0052] | WITHIN | 0.0047 | +0.0043 | +0.0143 |
| squad | in-domain | 11873 | 0.9116 | 0.9111 | -0.0005 [-0.0024, +0.0013] | WITHIN | 0.9054 | -0.0005 | +0.0017 |
| musique | zero-shot | 2417 | 0.5129 | 0.5114 | -0.0015 [-0.0061, +0.0030] | WITHIN | 0.4734 | -0.0029 | +0.0087 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9072 | +0.0031 [+0.0009, +0.0055] | WITHIN | 0.6846 | +0.0057 | -0.0073 |
| 2wiki | in-domain | 12576 | 0.9319 | 0.9375 | +0.0055 [+0.0040, +0.0072] | WITHIN | 0.6079 | +0.0153 | -0.0018 |
| webqsp | zero-shot | 1503 | 0.3570 | 0.3522 | -0.0047 [-0.0140, +0.0047] | WITHIN | 0.0535 | -0.0047 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
