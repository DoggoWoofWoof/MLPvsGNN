# Screen scr-u2d-new-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zg6 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8110 | 0.8134 | +0.0024 [+0.0012, +0.0035] | WITHIN | 0.0047 | +0.0028 | +0.0041 |
| squad | in-domain | 11873 | 0.9117 | 0.9141 | +0.0024 [+0.0004, +0.0043] | WITHIN | 0.9054 | +0.0024 | +0.0002 |
| musique | zero-shot | 2417 | 0.5414 | 0.5426 | +0.0013 [-0.0047, +0.0072] | WITHIN | 0.4734 | -0.0021 | +0.0434 |
| hotpotqa | in-domain | 7405 | 0.8571 | 0.8604 | +0.0034 [+0.0008, +0.0059] | WITHIN | 0.6846 | +0.0045 | +0.0166 |
| 2wiki | in-domain | 12576 | 0.9266 | 0.9306 | +0.0039 [+0.0023, +0.0057] | WITHIN | 0.6079 | +0.0087 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.1961 | 0.1560 | -0.0401 [-0.0558, -0.0237] | LOSS | 0.0535 | -0.0279 | -0.0479 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
