# Screen scr-u2h-zrckpw-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8033 | 0.8043 | +0.0011 [+0.0003, +0.0018] | WITHIN | 0.0047 | +0.0012 | +0.0030 |
| squad | in-domain | 11873 | 0.9124 | 0.9105 | -0.0019 [-0.0040, +0.0002] | WITHIN | 0.9054 | -0.0019 | -0.0002 |
| musique | in-domain | 2417 | 0.5697 | 0.5736 | +0.0039 [-0.0018, +0.0095] | WITHIN | 0.4734 | +0.0058 | -0.0074 |
| hotpotqa | zero-shot | 7405 | 0.8538 | 0.8469 | -0.0069 [-0.0099, -0.0037] | WITHIN | 0.6846 | -0.0126 | +0.0034 |
| 2wiki | in-domain | 12576 | 0.9303 | 0.9381 | +0.0078 [+0.0062, +0.0096] | GAIN | 0.6079 | +0.0218 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.3370 | 0.3312 | -0.0057 [-0.0160, +0.0048] | WITHIN | 0.0535 | -0.0086 | -0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
