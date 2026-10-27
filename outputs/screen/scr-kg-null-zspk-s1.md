# Screen scr-kg-base-zspk-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8038 | +0.0025 [+0.0015, +0.0036] | WITHIN | 0.0047 | +0.0031 | +0.0080 |
| squad | in-domain | 11873 | 0.9102 | 0.9109 | +0.0007 [-0.0012, +0.0025] | WITHIN | 0.9054 | +0.0007 | -0.0002 |
| musique | zero-shot | 2417 | 0.5146 | 0.5080 | -0.0066 [-0.0122, -0.0011] | WITHIN | 0.4734 | -0.0066 | +0.0062 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9075 | -0.0011 [-0.0030, +0.0009] | WITHIN | 0.6846 | -0.0031 | +0.0059 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9334 | -0.0030 [-0.0044, -0.0016] | WITHIN | 0.6079 | -0.0086 | +0.0064 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.3160 | +0.0009 [-0.0077, +0.0102] | WITHIN | 0.0535 | +0.0007 | -0.0113 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
