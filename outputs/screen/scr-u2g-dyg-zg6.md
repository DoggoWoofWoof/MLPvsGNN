# Screen scr-u2g-y-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zg6 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8123 | 0.8110 | -0.0013 [-0.0022, -0.0003] | WITHIN | 0.0047 | -0.0011 | -0.0024 |
| squad | in-domain | 11873 | 0.9106 | 0.9117 | +0.0011 [-0.0012, +0.0034] | WITHIN | 0.9054 | +0.0011 | +0.0014 |
| musique | zero-shot | 2417 | 0.5207 | 0.5414 | +0.0207 [+0.0136, +0.0282] | GAIN | 0.4734 | +0.0165 | -0.0252 |
| hotpotqa | in-domain | 7405 | 0.9118 | 0.8571 | -0.0548 [-0.0592, -0.0504] | LOSS | 0.6846 | -0.1111 | -0.0117 |
| 2wiki | in-domain | 12576 | 0.9416 | 0.9266 | -0.0149 [-0.0178, -0.0120] | LOSS | 0.6079 | -0.0372 | +0.0066 |
| webqsp | zero-shot | 1503 | 0.2368 | 0.1961 | -0.0407 [-0.0564, -0.0260] | LOSS | 0.0535 | -0.0353 | +0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
