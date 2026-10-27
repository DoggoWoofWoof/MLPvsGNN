# Screen scr-kg-base-zrckp-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8038 | 0.8023 | -0.0014 [-0.0025, -0.0003] | WITHIN | 0.0047 | -0.0009 | -0.0123 |
| squad | in-domain | 11873 | 0.9117 | 0.9127 | +0.0010 [-0.0009, +0.0029] | WITHIN | 0.9054 | +0.0010 | +0.0008 |
| musique | in-domain | 2417 | 0.5743 | 0.5713 | -0.0030 [-0.0083, +0.0024] | WITHIN | 0.4734 | +0.0000 | -0.0054 |
| hotpotqa | zero-shot | 7405 | 0.8504 | 0.8516 | +0.0012 [-0.0020, +0.0043] | WITHIN | 0.6846 | +0.0019 | +0.0099 |
| 2wiki | in-domain | 12576 | 0.9354 | 0.9354 | -0.0000 [-0.0015, +0.0015] | WITHIN | 0.6079 | -0.0011 | +0.0014 |
| webqsp | zero-shot | 1503 | 0.3289 | 0.3399 | +0.0110 [-0.0005, +0.0227] | WITHIN | 0.0535 | +0.0133 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
