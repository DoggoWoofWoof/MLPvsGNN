# Screen L-squad (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.6441 | -0.0094 [-0.0124, -0.0063] | LOSS | 0.0047 | -0.0128 | -0.0105 |
| squad | zero-shot | 11873 | 0.8797 | 0.8918 | +0.0120 [+0.0091, +0.0151] | GAIN | 0.9054 | +0.0120 | +0.0153 |
| musique | in-domain | 2417 | 0.5743 | 0.5616 | -0.0127 [-0.0184, -0.0066] | LOSS | 0.4734 | -0.0170 | -0.0132 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.8995 | -0.0049 [-0.0075, -0.0024] | WITHIN | 0.6846 | -0.0072 | +0.0132 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8711 | -0.0013 [-0.0033, +0.0006] | WITHIN | 0.6079 | -0.0010 | +0.0056 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.1310 | +0.0008 [-0.0087, +0.0094] | WITHIN | 0.0535 | -0.0013 | +0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
