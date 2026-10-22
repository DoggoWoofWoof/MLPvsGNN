# Screen scr-u2d-base-zrckp-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8033 | 0.8023 | -0.0010 [-0.0020, -0.0000] | WITHIN | 0.0047 | -0.0001 | -0.0080 |
| squad | in-domain | 11873 | 0.9124 | 0.9122 | -0.0003 [-0.0021, +0.0017] | WITHIN | 0.9054 | -0.0003 | +0.0030 |
| musique | in-domain | 2417 | 0.5697 | 0.5710 | +0.0013 [-0.0044, +0.0070] | WITHIN | 0.4734 | +0.0012 | +0.0046 |
| hotpotqa | zero-shot | 7405 | 0.8538 | 0.8492 | -0.0046 [-0.0077, -0.0016] | WITHIN | 0.6846 | -0.0082 | -0.0062 |
| 2wiki | in-domain | 12576 | 0.9303 | 0.9324 | +0.0021 [+0.0004, +0.0038] | WITHIN | 0.6079 | +0.0086 | -0.0017 |
| webqsp | zero-shot | 1503 | 0.3370 | 0.3534 | +0.0164 [+0.0057, +0.0274] | GAIN | 0.0535 | +0.0073 | +0.0073 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
