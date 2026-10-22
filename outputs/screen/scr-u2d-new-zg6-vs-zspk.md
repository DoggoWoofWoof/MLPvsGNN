# Screen scr-u2d-new-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-new-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7991 | 0.8134 | +0.0143 [+0.0124, +0.0162] | GAIN | 0.0047 | +0.0158 | +0.0478 |
| squad | in-domain | 11873 | 0.9137 | 0.9141 | +0.0004 [-0.0016, +0.0024] | WITHIN | 0.9054 | +0.0004 | +0.0015 |
| musique | zero-shot | 2417 | 0.5257 | 0.5426 | +0.0170 [+0.0104, +0.0230] | GAIN | 0.4734 | +0.0112 | -0.0029 |
| hotpotqa | in-domain | 7405 | 0.8483 | 0.8604 | +0.0121 [+0.0092, +0.0151] | GAIN | 0.6846 | +0.0203 | +0.0332 |
| 2wiki | in-domain | 12576 | 0.9187 | 0.9306 | +0.0119 [+0.0101, +0.0138] | GAIN | 0.6079 | +0.0349 | -0.0004 |
| webqsp | zero-shot | 1503 | 0.3223 | 0.1560 | -0.1663 [-0.1876, -0.1451] | LOSS | 0.0535 | -0.1251 | -0.1031 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
