# Screen scr-u2d-base-zrckp-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8026 | +0.0017 [+0.0007, +0.0029] | WITHIN | 0.0047 | +0.0017 | +0.0065 |
| squad | in-domain | 11873 | 0.9121 | 0.9116 | -0.0005 [-0.0024, +0.0014] | WITHIN | 0.9054 | -0.0005 | +0.0016 |
| musique | zero-shot | 2417 | 0.5153 | 0.5057 | -0.0096 [-0.0145, -0.0048] | LOSS | 0.4734 | -0.0153 | +0.0124 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.9049 | -0.0005 [-0.0026, +0.0015] | WITHIN | 0.6846 | +0.0005 | -0.0093 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9325 | -0.0020 [-0.0035, -0.0004] | WITHIN | 0.6079 | -0.0054 | +0.0048 |
| webqsp | zero-shot | 1503 | 0.3561 | 0.3596 | +0.0035 [-0.0054, +0.0132] | WITHIN | 0.0535 | +0.0040 | -0.0093 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
