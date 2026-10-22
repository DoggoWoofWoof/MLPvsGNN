# Screen scr-u2d-new-zspk-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8050 | 0.7999 | -0.0051 [-0.0067, -0.0036] | WITHIN | 0.0047 | -0.0060 | -0.0186 |
| squad | in-domain | 11873 | 0.9145 | 0.9159 | +0.0014 [-0.0005, +0.0035] | WITHIN | 0.9054 | +0.0014 | -0.0013 |
| musique | in-domain | 2417 | 0.5807 | 0.5765 | -0.0041 [-0.0098, +0.0012] | WITHIN | 0.4734 | -0.0012 | -0.0112 |
| hotpotqa | zero-shot | 7405 | 0.7966 | 0.8018 | +0.0052 [+0.0020, +0.0084] | WITHIN | 0.6846 | +0.0062 | +0.0154 |
| 2wiki | in-domain | 12576 | 0.9212 | 0.9095 | -0.0116 [-0.0136, -0.0098] | LOSS | 0.6079 | -0.0332 | -0.0029 |
| webqsp | zero-shot | 1503 | 0.2674 | 0.2942 | +0.0268 [+0.0132, +0.0399] | GAIN | 0.0535 | +0.0220 | +0.0180 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
