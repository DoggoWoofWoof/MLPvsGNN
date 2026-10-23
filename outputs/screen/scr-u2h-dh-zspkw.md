# Screen scr-u2h-zspkw (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.8033 | +0.0011 [+0.0003, +0.0019] | WITHIN | 0.0047 | +0.0011 | +0.0019 |
| squad | in-domain | 11873 | 0.9118 | 0.9122 | +0.0004 [-0.0013, +0.0023] | WITHIN | 0.9054 | +0.0004 | +0.0019 |
| musique | zero-shot | 2417 | 0.5143 | 0.5202 | +0.0059 [+0.0003, +0.0118] | WITHIN | 0.4734 | +0.0074 | -0.0070 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.9074 | -0.0015 [-0.0034, +0.0006] | WITHIN | 0.6846 | -0.0024 | -0.0109 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9369 | +0.0023 [+0.0009, +0.0038] | WITHIN | 0.6079 | +0.0036 | +0.0005 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.3192 | +0.0030 [-0.0053, +0.0104] | WITHIN | 0.0535 | +0.0027 | +0.0073 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
