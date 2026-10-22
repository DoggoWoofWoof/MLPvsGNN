# Screen scr-u2g-y-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8050 | 0.8117 | +0.0067 [+0.0053, +0.0081] | WITHIN | 0.0047 | +0.0083 | +0.0234 |
| squad | in-domain | 11873 | 0.9145 | 0.9132 | -0.0013 [-0.0035, +0.0008] | WITHIN | 0.9054 | -0.0013 | +0.0073 |
| musique | in-domain | 2417 | 0.5807 | 0.5984 | +0.0177 [+0.0115, +0.0237] | GAIN | 0.4734 | +0.0265 | +0.0149 |
| hotpotqa | zero-shot | 7405 | 0.7966 | 0.7847 | -0.0120 [-0.0162, -0.0078] | LOSS | 0.6846 | -0.0100 | -0.0379 |
| 2wiki | in-domain | 12576 | 0.9212 | 0.9271 | +0.0059 [+0.0040, +0.0078] | WITHIN | 0.6079 | +0.0207 | +0.0026 |
| webqsp | zero-shot | 1503 | 0.2674 | 0.2191 | -0.0483 [-0.0708, -0.0268] | LOSS | 0.0535 | -0.0413 | +0.0000 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
