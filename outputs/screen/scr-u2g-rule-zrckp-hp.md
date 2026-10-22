# Screen scr-u2d-new-zrckp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zrckp-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8046 | 0.8009 | -0.0037 [-0.0053, -0.0020] | WITHIN | 0.0047 | -0.0046 | -0.0125 |
| squad | in-domain | 11873 | 0.9146 | 0.9131 | -0.0015 [-0.0032, +0.0002] | WITHIN | 0.9054 | -0.0015 | -0.0035 |
| musique | in-domain | 2417 | 0.5766 | 0.5741 | -0.0026 [-0.0077, +0.0026] | WITHIN | 0.4734 | +0.0062 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8055 | 0.8064 | +0.0009 [-0.0026, +0.0046] | WITHIN | 0.6846 | +0.0016 | +0.0124 |
| 2wiki | in-domain | 12576 | 0.9218 | 0.9029 | -0.0189 [-0.0211, -0.0168] | LOSS | 0.6079 | -0.0485 | -0.0045 |
| webqsp | zero-shot | 1503 | 0.3369 | 0.3870 | +0.0501 [+0.0362, +0.0645] | GAIN | 0.0535 | +0.0446 | +0.0259 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
