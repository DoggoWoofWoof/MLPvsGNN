# Screen scr-r2t-zrckpb-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8038 | 0.8056 | +0.0018 [+0.0006, +0.0031] | WITHIN | 0.0047 | +0.0009 | +0.0050 |
| squad | in-domain | 11873 | 0.9117 | 0.9084 | -0.0033 [-0.0056, -0.0011] | WITHIN | 0.9054 | -0.0033 | -0.0067 |
| musique | in-domain | 2417 | 0.5743 | 0.5610 | -0.0133 [-0.0199, -0.0068] | LOSS | 0.4734 | -0.0116 | -0.0281 |
| hotpotqa | zero-shot | 7405 | 0.8504 | 0.8454 | -0.0050 [-0.0083, -0.0016] | WITHIN | 0.6846 | -0.0078 | -0.0117 |
| 2wiki | in-domain | 12576 | 0.9354 | 0.9340 | -0.0014 [-0.0030, +0.0002] | WITHIN | 0.6079 | -0.0007 | -0.0093 |
| webqsp | zero-shot | 1503 | 0.3289 | 0.3387 | +0.0098 [-0.0020, +0.0212] | WITHIN | 0.0535 | +0.0067 | +0.0086 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
