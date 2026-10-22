# Screen scr-u2g-y-zrckp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8009 | +0.0001 [-0.0008, +0.0011] | WITHIN | 0.0047 | +0.0000 | +0.0012 |
| squad | in-domain | 11873 | 0.9121 | 0.9132 | +0.0011 [-0.0011, +0.0034] | WITHIN | 0.9054 | +0.0011 | +0.0024 |
| musique | zero-shot | 2417 | 0.5153 | 0.5315 | +0.0163 [+0.0101, +0.0225] | GAIN | 0.4734 | +0.0132 | +0.0046 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.8512 | -0.0542 [-0.0588, -0.0497] | LOSS | 0.6846 | -0.1083 | +0.0031 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9171 | -0.0174 [-0.0204, -0.0146] | LOSS | 0.6079 | -0.0444 | +0.0051 |
| webqsp | zero-shot | 1503 | 0.3561 | 0.3523 | -0.0038 [-0.0146, +0.0073] | WITHIN | 0.0535 | -0.0033 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
