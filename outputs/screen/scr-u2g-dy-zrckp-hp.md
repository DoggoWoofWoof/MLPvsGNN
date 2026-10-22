# Screen scr-u2g-y-zrckp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8033 | 0.8046 | +0.0013 [+0.0005, +0.0022] | WITHIN | 0.0047 | +0.0021 | +0.0036 |
| squad | in-domain | 11873 | 0.9124 | 0.9146 | +0.0022 [-0.0000, +0.0045] | WITHIN | 0.9054 | +0.0022 | +0.0035 |
| musique | in-domain | 2417 | 0.5697 | 0.5766 | +0.0069 [+0.0007, +0.0131] | WITHIN | 0.4734 | +0.0012 | +0.0095 |
| hotpotqa | zero-shot | 7405 | 0.8538 | 0.8055 | -0.0483 [-0.0530, -0.0441] | LOSS | 0.6846 | -0.0943 | +0.0073 |
| 2wiki | in-domain | 12576 | 0.9303 | 0.9218 | -0.0085 [-0.0113, -0.0057] | LOSS | 0.6079 | -0.0195 | +0.0116 |
| webqsp | zero-shot | 1503 | 0.3370 | 0.3369 | -0.0001 [-0.0105, +0.0105] | WITHIN | 0.0535 | -0.0073 | +0.0093 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
