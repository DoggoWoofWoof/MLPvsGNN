# Screen scr-u2n-zrckpt-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8038 | 0.8039 | +0.0001 [-0.0008, +0.0010] | WITHIN | 0.0047 | +0.0004 | -0.0020 |
| squad | in-domain | 11873 | 0.9117 | 0.9128 | +0.0011 [-0.0008, +0.0030] | WITHIN | 0.9054 | +0.0011 | -0.0013 |
| musique | in-domain | 2417 | 0.5743 | 0.5762 | +0.0019 [-0.0031, +0.0069] | WITHIN | 0.4734 | +0.0017 | +0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8504 | 0.8583 | +0.0080 [+0.0051, +0.0108] | GAIN | 0.6846 | +0.0157 | +0.0077 |
| 2wiki | in-domain | 12576 | 0.9354 | 0.9349 | -0.0005 [-0.0019, +0.0011] | WITHIN | 0.6079 | +0.0001 | +0.0008 |
| webqsp | zero-shot | 1503 | 0.3289 | 0.3181 | -0.0108 [-0.0221, -0.0006] | LOSS | 0.0535 | -0.0080 | -0.0020 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
