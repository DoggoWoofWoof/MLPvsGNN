# Screen scr-a3z-zspkz (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8023 | +0.0014 [+0.0004, +0.0024] | WITHIN | 0.0047 | +0.0025 | +0.0098 |
| squad | in-domain | 11873 | 0.9114 | 0.9114 | +0.0000 [-0.0019, +0.0019] | WITHIN | 0.9054 | +0.0000 | +0.0003 |
| musique | zero-shot | 2417 | 0.5012 | 0.5195 | +0.0183 [+0.0129, +0.0237] | GAIN | 0.4734 | +0.0178 | +0.0046 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9047 | -0.0033 [-0.0055, -0.0009] | WITHIN | 0.6846 | -0.0068 | -0.0003 |
| 2wiki | in-domain | 12576 | 0.9383 | 0.9293 | -0.0089 [-0.0105, -0.0073] | LOSS | 0.6079 | -0.0222 | +0.0014 |
| webqsp | zero-shot | 1503 | 0.3229 | 0.3106 | -0.0122 [-0.0219, -0.0023] | LOSS | 0.0535 | -0.0073 | -0.0106 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8023 | +0.1486 [+0.1429, +0.1539] | GAIN | 0.0047 | +0.1478 | +0.2847 |
| squad | in-domain | 11873 | 0.9100 | 0.9114 | +0.0014 [-0.0007, +0.0035] | WITHIN | 0.9054 | +0.0014 | -0.0013 |
| musique | zero-shot | 2417 | 0.2696 | 0.5195 | +0.2499 [+0.2382, +0.2614] | GAIN | 0.4734 | +0.1341 | +0.3914 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9047 | +0.0032 [-0.0001, +0.0066] | WITHIN | 0.6846 | +0.0031 | +0.0047 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9293 | +0.0561 [+0.0530, +0.0594] | GAIN | 0.6079 | +0.1132 | +0.0737 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3106 | +0.1328 [+0.1103, +0.1574] | GAIN | 0.0535 | +0.1031 | +0.1171 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
