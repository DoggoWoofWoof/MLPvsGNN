# Screen scr-u2d-new-zrckp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zrckp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.7999 | -0.0010 [-0.0025, +0.0005] | WITHIN | 0.0047 | -0.0001 | -0.0054 |
| squad | in-domain | 11873 | 0.9132 | 0.9129 | -0.0003 [-0.0019, +0.0013] | WITHIN | 0.9054 | -0.0003 | +0.0003 |
| musique | zero-shot | 2417 | 0.5315 | 0.5189 | -0.0127 [-0.0191, -0.0065] | LOSS | 0.4734 | -0.0037 | +0.0033 |
| hotpotqa | in-domain | 7405 | 0.8512 | 0.8484 | -0.0028 [-0.0056, -0.0002] | WITHIN | 0.6846 | -0.0030 | -0.0096 |
| 2wiki | in-domain | 12576 | 0.9171 | 0.9204 | +0.0033 [+0.0017, +0.0051] | WITHIN | 0.6079 | +0.0083 | -0.0006 |
| webqsp | zero-shot | 1503 | 0.3523 | 0.3951 | +0.0428 [+0.0294, +0.0568] | GAIN | 0.0535 | +0.0386 | +0.0346 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
