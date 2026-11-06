# Screen scr-gnx-nx-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s1 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3477 | 0.3513 | +0.0036 [-0.0019, +0.0089] | WITHIN | 0.0047 | +0.0049 | +0.0233 |
| squad | in-domain | 11873 | 0.9145 | 0.9131 | -0.0014 [-0.0035, +0.0005] | WITHIN | 0.9054 | -0.0014 | -0.0035 |
| musique | zero-shot | 2417 | 0.5039 | 0.5208 | +0.0169 [+0.0109, +0.0231] | GAIN | 0.4734 | +0.0219 | +0.0099 |
| hotpotqa | in-domain | 7405 | 0.8596 | 0.8607 | +0.0011 [-0.0016, +0.0036] | WITHIN | 0.6846 | +0.0014 | -0.0038 |
| 2wiki | in-domain | 12576 | 0.9297 | 0.9283 | -0.0014 [-0.0031, +0.0004] | WITHIN | 0.6079 | -0.0059 | -0.0025 |
| webqsp | in-domain | 1503 | 0.6586 | 0.6661 | +0.0075 [-0.0022, +0.0167] | WITHIN | 0.0535 | +0.0093 | +0.0140 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
