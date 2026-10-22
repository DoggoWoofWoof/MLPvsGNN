# Screen scr-u2d-new-zspk (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.7991 | -0.0030 [-0.0046, -0.0015] | WITHIN | 0.0047 | -0.0025 | -0.0106 |
| squad | in-domain | 11873 | 0.9118 | 0.9137 | +0.0019 [-0.0007, +0.0044] | WITHIN | 0.9054 | +0.0019 | +0.0062 |
| musique | zero-shot | 2417 | 0.5143 | 0.5257 | +0.0113 [+0.0042, +0.0184] | GAIN | 0.4734 | +0.0153 | +0.0327 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.8483 | -0.0605 [-0.0652, -0.0558] | LOSS | 0.6846 | -0.1194 | -0.0224 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9187 | -0.0159 [-0.0188, -0.0131] | LOSS | 0.6079 | -0.0442 | +0.0073 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.3223 | +0.0060 [-0.0071, +0.0183] | WITHIN | 0.0535 | +0.0020 | +0.0140 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
